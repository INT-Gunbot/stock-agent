"""Agent 的工具集：把工作台的数据能力暴露成模型可调用的函数。

三条设计原则，决定了这个 Agent 能做什么、不能做什么：

1. **只读。** 所有工具不写库、不改 positions.csv、不下单。模型只能查和解释，
   动手的部分（更新数据、跑选股、保存持仓）留在按钮上，由人点。
2. **算术在工具里，不在模型里。** 均线、分位、盈亏都由 pandas 算好再交出去，
   模型只负责挑重点、组织语言。让模型自己算数是幻觉的主要来源。
3. **每个结果都带口径。** trade_date / as_of / source 一起返回，
   模型引用时不会把"收盘因子"说成"实时数据"。
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import pandas as pd

from src import indicators as I
from src import query as Q
from src import realtime as RT
from src.config import DATA_DIR, REPORT_DIR
from src.db import connect

BASE = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------- 小工具
def _con():
    return connect(read_only=True)


def _num(v, d=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return round(float(v), d)
    except (TypeError, ValueError):
        return None


def _load_daily_md() -> tuple[str, str] | tuple[None, None]:
    files = sorted(REPORT_DIR.glob("daily_*.md"))
    if not files:
        return None, None
    p = files[-1]
    return p.read_text(encoding="utf-8"), p.stem.replace("daily_", "")


def _positions_rows() -> list[dict]:
    p = BASE / "positions.csv"
    if not p.exists():
        return []
    return [dict(r) for r in csv.DictReader(io.StringIO(p.read_text(encoding="utf-8-sig")))]


def _latest_picks() -> dict | None:
    p = DATA_DIR / "recommendations.jsonl"
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = line
    return json.loads(last) if last else None


def _latest_checkup() -> dict | None:
    files = sorted(DATA_DIR.glob("checkup_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


_IND = {"at": 0.0, "map": {}}


def _industry_map(con) -> dict[str, str]:
    """ts_code -> 东财二级行业（128 个分类，覆盖 5580/5926 只）。

    来源是业绩报表里的行业字段，随财报一起入库，不用额外接口。
    缓存 5 分钟：这张映射一天最多变一次，但每次问答重建要扫 4 万行，不值得。
    """
    import time as _t
    if _IND["map"] and _t.time() - _IND["at"] < 300:
        return _IND["map"]
    rows = con.execute("""
        SELECT ts_code, arg_max(industry, end_date) AS ind
        FROM fina_yjbb WHERE industry IS NOT NULL GROUP BY 1
    """).fetchall()
    _IND.update(at=_t.time(), map={r[0]: r[1] for r in rows})
    return _IND["map"]


def _industry_of(con, code: str) -> str | None:
    return _industry_map(con).get(code)


def resolve_code(query: str) -> tuple[str | None, list[dict]]:
    """代码/名称 -> ts_code。名称模糊匹配可能命中多只，一并返回候选。"""
    s = str(query or "").strip().upper()
    if not s:
        return None, []
    if s[:6].isdigit():
        code = s[:6]
        if "." in s:
            return s, []
        con = _con()
        try:
            # ETF 先查：510300 这类代码不在股票后缀推断表里，硬猜会得到 .SZ，
            # 后面 kline/daily_bar 全查空，报的却是"找不到这只票"。
            row = con.execute(
                "SELECT ts_code, name FROM etf_dim WHERE ts_code LIKE ? LIMIT 1",
                [code + ".%"],
            ).fetchall()
            if not row:
                row = con.execute(
                    "SELECT ts_code, name FROM dim_stock WHERE ts_code LIKE ? LIMIT 1",
                    [code + ".%"],
                ).fetchall()
        finally:
            con.close()
        if row:
            return row[0][0], []
        # 查不到就别硬猜后缀。实测：831961 是北交所存量代码，2026 年已迁到
        # 920961，dim_stock 里只有新代码。硬拼成 831961.BJ 会一路查空，
        # 报出来却是"找不到这只票"，比"代码不存在"更容易误导。
        return None, []
    con = _con()
    try:
        # 东财的股票名里有空格（"五 粮 液"、"中 关 村"），用户不会带着空格输入，
        # 直接 LIKE 会漏。两边都去空格再比。
        kw = re.sub(r"[\s\u3000]+", "", str(query or "").strip().upper())
        rows = con.execute(
            "SELECT ts_code, name FROM etf_dim "
            "WHERE replace(replace(name, ' ', ''), '　', '') LIKE ? "
            "UNION ALL "
            "SELECT ts_code, name FROM dim_stock "
            "WHERE replace(replace(name, ' ', ''), '　', '') LIKE ? AND is_active "
            "LIMIT 8",
            [f"%{kw}%", f"%{kw}%"],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return None, []
    if len(rows) == 1:
        return rows[0][0], []
    return None, [{"ts_code": r[0], "name": r[1]} for r in rows]


def _asset_of(con, code: str) -> str:
    """ETF 走 etf_bar，股票走 daily_bar。以 etf_dim 为准，不按代码段猜。

    漏掉这一步的表现很隐蔽：ETF 的日线明明在库里，get_price 却按股票去查
    daily_bar，返回空表，工具对外说"无日线数据"——数据在、被判成不在。
    """
    try:
        row = con.execute(
            "SELECT 1 FROM etf_dim WHERE ts_code = ? LIMIT 1", [code]
        ).fetchone()
        return "etf" if row else "stock"
    except Exception:
        return "stock"


def _name_of(con, code: str) -> str:
    r = con.execute(
        "SELECT name FROM dim_stock WHERE ts_code = ? "
        "UNION ALL SELECT name FROM etf_dim WHERE ts_code = ?",
        [code, code],
    ).fetchall()
    return r[0][0] if r else code


# ---------------------------------------------------------------- 工具实现
_NOT_FOUND_HINT = ("本地股票池里没有这个代码或名称。若用的是北交所老代码，"
                   "存量代码 2026 年已迁到 920xxx 段（如创远信科 831961 → 920961），"
                   "改用股票名称再查一次。")

def market_now() -> dict:
    """市场现状：实时指数/涨跌家数 + 最近收盘日的档位判定。"""
    snap = RT.snapshot(with_breadth=True)
    out = {
        "口径": RT.market_label(snap),
        "交易时段": snap["session"]["text"],
        "指数": [
            {"名称": i["name"], "点位": _num(i["close"]),
             "涨跌幅%": _num(i["pct"]),
             "昨收": bool(i.get("pre")),
             "成交额亿": _num((i["amount"] or 0) / 1e8, 0) if i.get("amount") else None}
            for i in snap["indices"]
        ],
        "涨跌家数": None,
        "市场档位": None,
    }
    if snap.get("pre_open"):
        out["提示"] = "盘前尚未开盘：指数点位是上一交易日收盘，没有当日涨跌幅"
    b = snap.get("breadth")
    if b:
        out["涨跌家数"] = {
            "上涨": b["up"], "下跌": b["down"], "平盘": b["flat"],
            "涨停": b["limit_up"], "跌停": b["limit_down"],
            "封板率%": b["seal_rate"],
            "全市场成交额亿": _num((b["amount"] or 0) / 1e8, 0),
        }
    try:
        from src.market import build as market_build
        from src.query import latest_trade_date
        con = _con()
        try:
            td = latest_trade_date(con)
            mk = market_build(con, td)
        finally:
            con.close()
        rg = mk.get("regime") or {}
        out["市场档位"] = {
            "截止日": str(td),
            "档位": rg.get("level"),
            "依据": rg.get("reason"),
            "建议仓位上限%": rg.get("position_cap"),
            "三因子权重": _weights(rg.get("level")),
        }
        out["上一交易日成交额亿"] = _num((mk.get("turnover", {}).get("amount") or 0) / 1e8, 0)
    except Exception as e:
        out["市场档位"] = {"error": f"数据库不可读（可能正在更新）：{str(e)[:80]}"}
    if snap.get("note"):
        out["注意"] = snap["note"]
    return out


def portfolio_review(focus: str | None = None) -> dict:
    """持仓体检：实时价 + 复权盈亏 + 分层建议 + 待处理事项。"""
    ck = _latest_checkup()
    rows = _positions_rows()
    if not ck and not rows:
        return {"error": "没有持仓数据"}
    codes = [r.get("ts_code", "").strip() for r in rows if r.get("ts_code")]
    q = RT.quotes(codes) if codes else {}
    cost_map = {r["ts_code"].strip(): r for r in rows if r.get("ts_code")}

    hold_list = []
    live_mv = live_day = 0.0
    for h in (ck or {}).get("holdings", []):
        code = h.get("ts_code")
        qq = q.get(code) or {}
        pos = cost_map.get(code) or {}
        try:
            shares = float(pos.get("shares") or 0)
        except ValueError:
            shares = 0.0
        price = qq.get("price") or h.get("close")
        mv = (price or 0) * shares
        pre = qq.get("pre_close")
        day = (price - pre) * shares if (price and pre) else None
        live_mv += mv
        live_day += day or 0
        item = {
            "名称": h.get("name"), "代码": code,
            "你的标注": h.get("tag") or "未标注",
            "建议仓别": h.get("suggest"),
            "标注冲突": bool(h.get("tag_conflict")),
            "实时价": _num(price), "今日涨跌%": _num(qq.get("pct")),
            "复权盈亏%": _num(h.get("pnl_pct_adj"), 1),
            "仓位占比%": _num(h.get("weight"), 1),
            "20日位置": _num(h.get("pos20")),
            "MA20上方": h.get("above_ma20"), "MA60上方": h.get("above_ma60"),
        }
        if item["今日涨跌%"] is None:
            item["今日涨跌%"] = _num(h.get("day_ret"))
        hold_list.append(item)

    prof = (ck or {}).get("profile") or {}
    out = {
        "持仓只数": prof.get("positions") or len(hold_list),
        "实时市值": _num(live_mv, 0),
        "今日盈亏": _num(live_day, 0),
        "收盘口径总盈亏": _num(prof.get("pnl"), 0),
        "收盘口径总盈亏%": _num(prof.get("pnl_pct"), 2),
        "当日涨跌%": _num(prof.get("day_ret"), 2),
        "集中度": {
            "前三大合计占比%": _num(prof.get("top3_weight"), 1),
            "最大单一持仓": prof.get("max_weight_name"),
            "最大单一占比%": _num(prof.get("max_weight"), 1),
            "单一持仓上限%": prof.get("concentration_limit"),
            "最大单一是否超限": (prof.get("max_weight") or 0) > (prof.get("concentration_limit") or 999),
            "说明": "上限针对单只，不是前三大合计",
        },
        "档位": prof.get("regime"), "建议仓位上限%": prof.get("position_cap"),
        "持仓明细": hold_list,
        "待处理": [
            {"名称": a.get("name"), "建议动作": a.get("action"),
             "紧急度": {3: "今天", 2: "本周", 0: "观察"}.get(a.get("severity"), "观察"),
             "原因": a.get("why")}
            for a in (ck or {}).get("actions", [])
        ],
        "口径": "盈亏% 为复权口径（分红送转不会误判）；市值与今日盈亏用实时价",
        "仓位数据缺口": ("positions.csv 只记持仓，没有现金和总资产，算不出真实仓位比例。"
                     "「建议仓位上限%」只能当作持仓规模的约束参考，回答仓位问题时必须说明这一点"),
    }
    if focus:
        code, cands = resolve_code(focus)
        if cands:
            return {"需要确认": f"「{focus}」匹配到多只，代码是？", "候选": cands}
        hit = [x for x in hold_list if x["代码"] == code]
        out["聚焦"] = hit[0] if hit else {"提示": f"{focus} 不在当前持仓里"}
    return out


def stock_lookup(query: str) -> dict:
    """单只股票：实时价 + 技术位置 + 最新财报 + 是否持有/在榜。"""
    code, cands = resolve_code(query)
    if cands:
        return {"需要确认": f"「{query}」匹配到多只，请指定代码", "候选": cands}
    if not code:
        return {"error": f"找不到「{query}」", "提示": _NOT_FOUND_HINT}

    out: dict = {"代码": code}
    qq = RT.quotes([code]).get(code) or {}
    if qq:
        out["名称"] = qq.get("name")
        out["实时价"] = _num(qq.get("price"))
        out["今日涨跌%"] = _num(qq.get("pct"))
        out["今开/最高/最低"] = [_num(qq.get("open")), _num(qq.get("high")), _num(qq.get("low"))]
        out["昨收"] = _num(qq.get("pre_close"))
        out["量比"] = _num(qq.get("vol_ratio"))
        out["换手率%"] = _num(qq.get("turnover"))
        out["是否停牌"] = qq.get("suspended")
        out["今日涨跌停幅度%"] = qq.get("limit_pct")
        if qq.get("no_quote"):
            out["报价状态"] = "盘前/休市无当日报价，实时价为空，以下判断用收盘日线口径"
    else:
        out["注意"] = "实时报价未取到，以下为收盘数据"

    con = _con()
    try:
        asset = _asset_of(con, code)
        if not out.get("名称"):
            out["名称"] = _name_of(con, code)
        if asset == "etf":
            # rules.limit_pct 只按板块分（主板 10%/双创 20%/北交所 30%），
            # 对 ETF 会返回 10% —— 而 ETF 本身没有涨跌幅限制（跨境 10% 是
            # 基金层面的约定，不构成"涨停封板"）。不纠正的话，模型会拿这个
            # 10% 去解释 ETF 的涨幅，得出"接近涨停"的错误结论。
            out["今日涨跌停幅度%"] = None
            out["涨跌停说明"] = ("ETF 没有涨跌幅限制（跨境 ETF 为 10%），"
                             "不适用个股的涨停/跌停判定")
        px = Q.get_price(code, adjust="hfq", con=con, asset=asset)
        if px.empty:
            out["行情"] = "无日线数据"
        else:
            px = I.add_all(Q.attach_return(px))
            last = px.iloc[-1]
            f_now = float(last["adj_factor"]) or 1.0
            # 复权价 -> 现价口径，否则茅台 MA20 会显示成 11700
            back = lambda v: _num(float(v) / f_now) if pd.notna(v) else None  # noqa: E731
            adj_close = float(last["adj_close"])
            out["日线截止"] = str(last["trade_date"])[:10]
            out["收盘价"] = _num(last["close"])
            out["均线"] = {f"MA{n}": back(last.get(f"ma{n}")) for n in (5, 10, 20, 60)}
            out["MA20上方"] = bool(pd.notna(last.get("ma20")) and adj_close > last["ma20"])
            out["MA60上方"] = bool(pd.notna(last.get("ma60")) and adj_close > last["ma60"])
            out["20日位置"] = _num(last.get("pos20"))
            out["60日位置"] = _num(last.get("pos60"))
            out["区间涨跌%"] = {
                "5日": _num((last.get("ret5") or 0) * 100, 1) if pd.notna(last.get("ret5")) else None,
                "20日": _num((last.get("ret20") or 0) * 100, 1) if pd.notna(last.get("ret20")) else None,
                "60日": _num((last.get("ret60") or 0) * 100, 1) if pd.notna(last.get("ret60")) else None,
            }
            out["20日最高/最低"] = [back(last.get("hi20")), back(last.get("lo20"))]
            out["量比/ATR%"] = [_num(last.get("vol_ratio")), _num(last.get("atr_pct"))]
            out["近日除权"] = bool(last.get("is_exdiv", False))

        f = Q.get_fina_latest(code, as_of=None, con=con) or {}
        if f:
            out["行业"] = _industry_of(con, code)
            out["最新财报"] = {
                "报告期": str(f.get("end_date", ""))[:10],
                "可见日": str(f.get("visible_date", ""))[:10],
                "ROE%": _num(f.get("roe")),
                "营收同比%": _num(f.get("revenue_yoy")),
                "净利同比%": _num(f.get("net_profit_yoy")),
                "净利润(亿)": _num((f.get("net_profit") or 0) / 1e8 if f.get("net_profit") is not None else None),
            }
        else:
            out["最新财报"] = "无（未披露或被剔除）"
    except Exception as e:
        out["数据库"] = f"读取失败：{str(e)[:80]}"
    finally:
        con.close()

    # 持仓与榜单的上下文，模型回答"还能拿吗"时必须有
    pos = {r["ts_code"].strip(): r for r in _positions_rows() if r.get("ts_code")}
    if code in pos:
        r = pos[code]
        try:
            sh, cst = float(r.get("shares") or 0), float(r.get("cost") or 0)
            price = out.get("实时价") or out.get("收盘价")
            out["我的持仓"] = {
                "股数": sh, "成本": cst, "买入日": r.get("open_date"),
                "标注": r.get("tag"), "备注": r.get("note"),
                "浮动盈亏%": _num((price / cst - 1) * 100, 2) if (price and cst) else None,
            }
        except ValueError:
            pass
    picks = _latest_picks()
    if picks:
        rank = next((i + 1 for i, p in enumerate(picks["picks"]) if p["ts_code"] == code), None)
        out["选股榜"] = {"榜单日期": picks["trade_date"], "名次": rank} if rank else \
            {"榜单日期": picks["trade_date"], "名次": "未入榜"}
    return out


def stock_history(query: str, days: int = 20) -> dict:
    """近 N 个交易日：逐日涨跌、量比，以及区间统计。"""
    code, cands = resolve_code(query)
    if cands:
        return {"需要确认": f"「{query}」匹配到多只", "候选": cands}
    if not code:
        return {"error": f"找不到「{query}」", "提示": _NOT_FOUND_HINT}
    days = max(5, min(int(days or 20), 60))
    con = _con()
    try:
        asset = _asset_of(con, code)
        px = Q.get_price(code, adjust="hfq", con=con, asset=asset)
        if px.empty:
            return {"error": "无日线数据"}
        px = Q.attach_return(px).tail(days)
        name = _name_of(con, code)
        bars = [
            {"日期": str(r["trade_date"])[:10], "收盘": _num(r["close"]),
             "涨跌%": _num((r["ret"] or 0) * 100, 2) if pd.notna(r.get("ret")) else None,
             "成交额亿": _num((r["amount"] or 0) / 1e8, 1)}
            for _, r in px.iterrows()
        ]
        close = px["close"].astype(float)
        return {
            "名称": name, "代码": code,
            "类型": "ETF" if asset == "etf" else "股票",
            "区间": f"{bars[0]['日期']} ~ {bars[-1]['日期']}",
            "逐日": bars,
            "区间统计": {
                "累计涨跌%": _num((close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
                "最高": _num(close.max()), "最低": _num(close.min()),
                "上涨天数": int((px["ret"] > 0).sum()), "下跌天数": int((px["ret"] < 0).sum()),
            },
            "口径": "价格不复权，涨跌% 为复权收益率（除权日不失真）",
        }
    finally:
        con.close()


def kline_view(query: str, period: str = "day", days: int = 120) -> dict:
    """K 线视角的走势画像。

    刻意复用 src/kline.py —— 页面上那张图和这里给模型看的数字是同一份计算，
    否则会出现"图上涨了 10%，Agent 说涨了 12%"这种最难排查的不一致。
    和 stock_history 的分工：那个给逐日流水，这个给形态与统计（回撤、波动、
    均线排列、量能），用于回答"最近走得怎么样""是不是放量突破"。
    """
    from src import kline as K

    period = period if period in K.PERIODS else "day"
    n = max(20, min(int(days or 120), 750))
    r = K.build(query, n=n, period=period, adjust="qfq")
    if r.get("error"):
        return {"error": r["error"], "候选": r.get("candidates") or []}

    c, v = r["c"], r["v"]
    ma = r["ma"]
    last = {w: (ma[str(w)][-1] if ma.get(str(w)) else None) for w in K.MA_WINDOWS}
    px = c[-1]
    # 均线排列：多头/空头/纠缠。判断依据是四条均线的相对次序，不看到数值来回猜。
    vals = [last[w] for w in K.MA_WINDOWS]
    if all(vals):
        if vals[0] > vals[1] > vals[2] > vals[3]:
            order = "多头排列（MA5>MA10>MA20>MA60）"
        elif vals[0] < vals[1] < vals[2] < vals[3]:
            order = "空头排列（MA5<MA10<MA20<MA60）"
        else:
            order = "交织（均线未形成单向排列）"
    else:
        order = "样本不足，无法判断排列"

    # 量能：最近一根对比此前 5 根均量。用已收盘的根，盘中最后一根量不完整要说明。
    base = [x for x in v[-6:-1] if x]
    vol_ratio = round(v[-1] / (sum(base) / len(base)), 2) if base and v[-1] else None

    s = r["stats"]
    out = {
        "名称": r["name"], "代码": r["code"],
        "类型": "ETF" if r.get("asset") == "etf" else "股票",
        "周期": r["period_label"], "复权": "前复权",
        "区间": f"{s['首日']} ~ {s['末日']}（{s['根数']} 根）",
        "最新价": round(px, 2),
        "区间涨跌%": s["区间涨跌"],
        "最大回撤%": s["最大回撤"],
        "年化波动%": s["年化波动"],
        "距区间高点%": s["距高点"],
        "区间最高": s["区间最高"], "区间最低": s["区间最低"],
        "均线": {f"MA{w}": last[w] for w in K.MA_WINDOWS},
        "均线排列": order,
        "现价位置": {
            "站上MA20": None if not last[20] else px > last[20],
            "站上MA60": None if not last[60] else px > last[60],
        },
        "最近一根量能比": vol_ratio,
        "最后一根": {"日期": r["t"][-1], "涨跌%": r["pct"][-1],
                    "是否盘中": bool(r["is_live_last"])},
        "口径": ("前复权，以最新交易日为基准；均线在所选周期上计算；"
                 "最大回撤与波动按区间内收盘价算。盘中的最后一根未收盘，"
                 "涨跌幅和成交量还会变。"),
        "看图入口": f"工作台「K线」标签页，输入 {r['code']} 可看同一条数据的图。",
    }
    if r["is_live_last"] and vol_ratio is not None:
        out["量能说明"] = ("最后一根是盘中数据，成交量还没走完，量比会偏低，"
                       "不要据此断言放量或缩量")
    if r["note"]:
        out["除权提示"] = r["note"]
    if r.get("position"):
        p = r["position"]
        out["我的持仓"] = {
            "股数": p["shares"], "成本": p["cost"],
            "复权盈亏%": p["pnl_pct"], "浮动盈亏": p["pnl"],
            "标注": p["tag"] or None, "买入日": p["open_date"] or None,
        }
    return out


def _industry_from_db(members: list[str]) -> tuple[list[dict], str]:
    """盘前/休市时用上一交易日收盘数据算行业强弱。

    用中位数而不是均值：除权日的 pct_chg 会失真（10 转 9 当天显示 -47%），
    中位数对少数异常值不敏感，比"把均值洗干净"更省事也更稳。
    """
    if not members:
        return [], ""
    con = _con()
    try:
        td = Q.latest_trade_date(con)
        ph = ",".join("?" * len(members))
        df = con.execute(
            f"""SELECT b.ts_code, d.name, b.pct_chg, b.close, b.amount
                FROM daily_bar b JOIN dim_stock d USING (ts_code)
                WHERE b.trade_date = ? AND b.ts_code IN ({ph})""",
            [td] + members,
        ).df()
    finally:
        con.close()
    if df.empty:
        return [], ""
    rows = []
    for r in df.itertuples(index=False):
        pct = _num(r.pct_chg)
        if pct is None:
            continue
        # 超过该股涨跌停幅度的，基本是除权失真，剔掉
        if abs(pct) > RT._limit_pct(r.ts_code.split(".")[0], str(r.name)) + 1:
            continue
        rows.append({"代码": r.ts_code, "名称": r.name, "涨跌%": pct,
                     "价格": _num(r.close), "成交额亿": _num((r.amount or 0) / 1e8, 1)})
    return rows, f"上一交易日（{td}）收盘口径"


def industry_view(query: str) -> dict:
    """行业视图：该股所属行业今天的整体强弱，以及它在本行业里的位置。

    行业映射来自业绩报表的「行业」字段（东财二级行业，128 个分类），
    不是申万分类——颗粒度比申万一级细、比三级粗，回答"哪个板块强"够用，
    做严格的行业中性化不够用，这点要跟用户讲清楚。
    """
    con = _con()
    try:
        imap = _industry_map(con)
    finally:
        con.close()
    if not imap:
        return {"error": "行业映射为空，需要先跑一次财务数据更新"}

    code, cands = resolve_code(query)
    if cands:
        return {"需要确认": f"「{query}」匹配到多只", "候选": cands}

    ind = imap.get(code) if code else None
    if not ind:
        q = str(query).replace("Ⅱ", "").strip()
        hits = [n for n in set(imap.values()) if q and (q in n or n.replace("Ⅱ", "") in q)]
        if not hits:
            return {"error": f"没找到「{query}」对应的行业"}
        ind = max(hits, key=lambda n: sum(1 for v in imap.values() if v == n))

    members = [c for c, v in imap.items() if v == ind][:400]
    live = RT.quotes(members)
    rows, basis = [], "盘中实时"
    for c in members:
        qq = live.get(c) or {}
        if qq.get("pct") is None:
            continue
        rows.append({"代码": c, "名称": qq.get("name"), "涨跌%": _num(qq.get("pct")),
                     "价格": _num(qq.get("price")), "成交额亿": _num((qq.get("amount") or 0) / 1e8, 1)})
    if not rows:
        # 盘前 / 休市：实时接口没有当日涨跌，退到上一交易日收盘口径，别让用户问了个寂寞
        rows, basis = _industry_from_db(members)
        if not rows:
            return {"行业": ind, "成员数": len(members), "提示": "成分股报价与日线都未取到"}
    rows.sort(key=lambda r: r["涨跌%"], reverse=True)
    pcts = [r["涨跌%"] for r in rows]
    up = sum(1 for p in pcts if p > 0)
    me = next((i + 1 for i, r in enumerate(rows) if r["代码"] == code), None)
    return {
        "行业": ind,
        "口径": basis,
        "成员数": len(rows),
        "平均涨跌%(中位数)": _num(pd.Series(pcts).median(), 2),
        "均值涨跌%": _num(sum(pcts) / len(pcts), 2),
        "上涨/下跌": f"{up} / {len(pcts) - up}",
        "成交额合计亿": _num(sum(r["成交额亿"] or 0 for r in rows), 0),
        "涨幅前5": rows[:5],
        "跌幅前5": rows[-5:][::-1],
        "本股行业排位": f"{me} / {len(rows)}" if me else None,
        "说明": "口径是「" + basis + "」。平均涨跌取中位数不用均值——除权失真会把均值带偏。"
                "行业分类是东财二级（128 类），不是申万",
    }


def top_picks(n: int = 10) -> dict:
    """当前选股榜单 + 打分明细 + 否决统计。榜单是观察池，不是买入指令。"""
    picks = _latest_picks()
    if not picks:
        return {"error": "还没有跑过选股，请先点「跑选股」"}
    n = max(1, min(int(n or 10), 20))
    rows = picks["picks"][:n]
    live = RT.quotes([p["ts_code"] for p in rows])
    out = []
    for i, p in enumerate(rows, 1):
        qq = live.get(p["ts_code"]) or {}
        out.append({
            "名次": i, "代码": p["ts_code"], "名称": p["name"],
            "实时价": _num(qq.get("price")) or _num(p.get("close")),
            "今日涨跌%": _num(qq.get("pct")),
            "综合分": _num(p.get("score"), 3),
            "资金面": _num(p.get("score_A"), 3),
            "基本面": _num(p.get("score_B"), 3),
            "技术位置": _num(p.get("score_C"), 3),
            "20日涨跌%": _num((p.get("ret20") or 0) * 100, 1),
            "量比": _num(p.get("vol_ratio")),
            "ROE%": _num(p.get("roe")), "净利同比%": _num(p.get("net_profit_yoy")),
            "入选理由": p.get("reason"),
        })
    v = picks.get("veto") or {}
    return {
        "榜单日期": picks["trade_date"], "市场档位": picks["regime"],
        "池子": f'{picks.get("universe")} 只 -> 可用 {picks.get("pool")} 只',
        "因子权重": _weights(picks.get("regime")),
        "榜单": out,
        "否决明细": {k: v for k, v in v.items() if k not in ("初始池", "可用池") and v},
        "口径": "因子基于收盘日线；实时价与今日涨跌为盘中数据。资金面是量价代理，未接真实资金流。",
    }


def _weights(regime: str | None) -> dict:
    table = {
        "进攻": {"资金面": 45, "基本面": 25, "技术位置": 30},
        "震荡": {"资金面": 40, "基本面": 35, "技术位置": 25},
        "防守": {"资金面": 25, "基本面": 50, "技术位置": 25},
    }
    return table.get(str(regime), {})


def daily_digest() -> dict:
    """最新日报的核心结论（今日要点、异动）。"""
    md, d = _load_daily_md()
    if not md:
        return {"error": "还没有日报，请先运行「更新数据 + 日报」"}
    lines = md.splitlines()
    out: dict = {"日报日期": d}
    try:
        i = lines.index("**今日要点**")
        pts = []
        for ln in lines[i + 1:]:
            s = ln.strip()
            if s.startswith("- "):
                pts.append(s[2:].strip())
            elif s:
                break
        out["今日要点"] = pts
    except ValueError:
        pass
    for key in ("异动", "持仓"):
        hdr = [x for x in lines if x.startswith(f"## {key}")]
        if hdr:
            idx = lines.index(hdr[0])
            body = []
            for ln in lines[idx + 1:]:
                if ln.startswith("## "):
                    break
                if ln.strip() and not ln.strip().startswith("|"):
                    body.append(ln.strip())
            out[key] = body[:6]
    out["全文"] = md[:2600]
    return out


# 时效性诉求的识别。只用于给 rag_search 加一条"再搜一次"的提示，不参与任何过滤，
# 所以宁可宽一点——误提示的代价是多一次 0.2 秒的搜索，漏提示的代价是模型
# 把"本地 8 天内没有"当成"最近没有"，那是错误结论。
_TIME_SENSITIVE = re.compile(
    r"最近|最新|近期|今年|本月|上周|这周|这几天|这几天|目前|现在有什么|"
    r"有没有.{0,4}(案例|消息|发生|出台|发布)|什么情况|什么进展")


DATA_GAPS = [
    "行业只有东财二级分类（128 类，来自业绩报表，覆盖 5580/5926 只）：能做粗的板块比较，"
    "没有申万三级、没有概念板块，也无法做行业中性化",
    "真实资金流（主力净流入、龙虎榜）未接入：因子里的「资金面」是量价代理",
    "北向资金、两融余额未接入",
    "质押、解禁、减持、股东户数未接入：无法判断筹码风险",
    "消息面没有入库、更没有结构化因子：能联网搜媒体摘要（web_search），"
    "但无法量化「某条消息对股价的冲击」，也无法批量筛「消息面利好」的票",
    "财务数据是季度披露的 point-in-time，不可能实时；当前用 min(公告日, 法定披露截止日) 做可见性判定",
    "positions.csv 没有现金和总资产，算不出真实仓位比例，只能给市值和占比",
]


def data_gaps() -> dict:
    """已知数据缺口。模型必须据此声明「不知道」，不许编。"""
    return {
        "缺口": DATA_GAPS,
        "已知限制": [
            "盘中实时价只用于估值与涨跌，因子/均线/榜单仍是收盘日线口径",
            "持仓盈亏的复权口径依赖买入日因子；买入日早于数据起点时按区间首日估算",
            "positions.csv 里的成本和股数由用户手填，系统不校验",
            "公告正文只覆盖 P0 类型（减持/回购/定增/诉讼/业绩预告等）并抽成关键段落，"
            "P2 类型（法律意见书、审计报告、独董述职、年报全文）不抓正文；"
            "关键段落是抽取不是全文，问极细的条款原文时要说明「摘要里没有，需看公告原文」",
        ],
    }


def rule_lookup(query: str) -> dict:
    """查本地规则卡：交易所规则与制度参数。

    和 rag_search 是两回事，不要混：rag_search 查的是「这个系统怎么做、之前怎么处理、
    哪家公司发了什么公告」；rule_lookup 查的是「市场的规则本身是什么」。
    前者是私有的、会变的；后者是公开的、有版本号的。

    为什么规则不能走 rag_search：实测问「ST 股票涨跌幅限制」，
    rag_search 命中「卫宁健康回购股份进展公告」并给出「高可信度」——
    规则类问句的特征词恰好是公告里的高频词。那不是拒答，是自信地答错。
    """
    from src import rules

    r = rules.lookup(query)
    if not r.get("命中"):
        return {
            "命中": 0,
            "结论": r.get("结论"),
            "提示": ("规则卡没收录，别再换关键词重试了。分两种情况处理："
                     "问概念或市场常识（PE 怎么算、定增与配股的区别）——直接答，"
                     "说明这是通用知识而非本系统数据；"
                     "问本系统的实现与口径（复权怎么处理、为什么不用实时价算因子）——"
                     "改用 rag_search 查方法论文档。"),
        }
    return r


def etf_screen(category: str | None = None, keyword: str | None = None,
               sort: str = "amt", min_amount: float = 500.0,
               max_premium: float | None = None, limit: int = 12) -> dict:
    """ETF 筛选。和个股选股是两套口径，不能互相替代。

    个股评分吃 ROE / 净利同比 / 毛利率 / PE 分位，ETF 这四样全部没有；
    反过来 ETF 有折溢价、IOPV、份额这些个股没有的东西。所以：
    问「哪只 ETF」用这个，问「哪只股票」用 top_picks，别串。
    """
    from src import etf as E

    r = E.screen(category=category, kw=keyword, sort=sort,
                 min_amount=min_amount, max_premium=max_premium,
                 limit=max(1, min(int(limit or 12), 40)))
    if r.get("error") or r.get("错误"):
        return r

    rows = []
    for x in r.get("行", []):
        rows.append({
            "代码": x["代码"], "名称": x["名称"], "分类": x["分类"],
            "现价": x["现价"], "今日%": x["今日%"], "溢价%": x["溢价%"],
            "日均成交亿": round(x["日均成交万"] / 1e4, 1) if x["日均成交万"] else None,
            "规模亿": x["规模亿"], "20日%": x["20日%"],
        })

    hi = [x for x in rows if (x["溢价%"] or 0) >= 3]
    warn = ""
    if hi:
        warn = (f"其中 {len(hi)} 只溢价超过 3%（{hi[0]['名称']} {hi[0]['溢价%']}%）。"
                "跨境 ETF 受外汇额度限制常年高溢价，溢价买入等于当场付掉这部分成本，"
                "指数不动也在亏。这是必须主动提示用户的事。")

    return {
        "命中": r.get("命中"),
        "排序": r.get("排序"),
        "条件": r.get("条件"),
        "列表": rows,
        "警示": warn or None,
        "说明": ("ETF 无涨跌幅限制，套涨跌停规则会得出错误结论；"
                 "分类按名称关键词推断，未识别的归入「其他」；"
                 "债券 ETF（511010 这类）不在当前数据源口径内，不在库里。"
                 "这是观察列表，不是推荐或买入指令。"),
    }


def rag_search(query: str, kind: str | None = None,
               code: str | None = None, days: int | None = None, k: int = 5) -> dict:
    """检索方法论文档、历史报告与公告。

    doc_key 里的文档是"这个系统为什么这么设计"的唯一权威来源：
    为什么用复权口径、为什么不能用实时价算因子、PIT 怎么判定、哪些数据没接入。
    这些问题答错就是编造，所以必须查文档而不是靠模型发挥。
    """
    from src.agent import rag as R

    try:
        r = R.search(query, k=min(int(k or 5), 8), kind=kind,
                     ts_code=code, days=days)
    except Exception as e:
        return {"error": f"检索失败：{type(e).__name__}: {str(e)[:120]}"}

    hits = r.get("结果") or []
    st = r.get("统计") or {}
    conf = st.get("可信度") or {}
    code_note = None

    if not hits and code:
        # 代码过滤命中 0 时不要直接宣告"没查到"。实测踩过：问创远信科的重组公告，
        # 模型传 831961（训练数据里的旧代码），但北交所存量代码已经迁到 920xxx 段
        # （831961 → 920961），dim_stock 里根本没有 831961，一条都过滤不出来，
        # 而公告其实就在库里、标题里还写着"创远信科"。
        # 所以退一层：去掉 ts_code 只按文本检索；再不行才去掉日期窗。
        for drop in ("ts_code", "days"):
            if drop == "ts_code":
                r2 = R.search(query, k=min(int(k or 5), 8), kind=kind, days=days)
            else:
                r2 = R.search(query, k=min(int(k or 5), 8), kind=kind)
            if r2.get("结果"):
                hits = r2["结果"]
                st = r2.get("统计") or {}
                conf = st.get("可信度") or {}
                code_note = (f"代码 {code} 在本地库里没有对应记录，已退回"
                             f"{'全文' if drop == 'ts_code' else '不限日期的'}检索。"
                             f"北交所存量代码已迁到 920xxx 段（如 831961 → 920961），"
                             f"代码对不上时属于这种情况，不要据此说「没有这条公告」。")
                break
        else:
            code_note = f"代码 {code} 过滤后无结果，去掉代码限制重查也没有。"

    if not hits:
        # 原来这里写的是「不要根据常识补全」，对所有问题一视同仁。
        # 但问「什么是市盈率」时这句话是错的——那是通识，本来就该直接用模型知识答，
        # 硬要求它说"没查到"会把一个能答的问题答成拒答。所以分开说。
        return {
            "命中": 0,
            "可信度": conf or {"等级": "无"},
            "结论": ("本地库里没有匹配内容。如果用户问的是本系统的实现细节、口径、或某份历史报告，"
                   "就说「文档里没有查到」；如果问的是概念定义或市场常识（PE 怎么算、定增与配股的区别），"
                   "那本来就不该查库，直接按通用知识回答，并注明这是通用知识而非本系统数据。"),
            "索引规模": st.get("块总数"),
            **({"代码说明": code_note} if code_note else {}),
        }

    out = []
    for h in hits:
        item = {
            "来源": h["source"],
            "类型": R.KIND_LABEL.get(h["kind"], h["kind"]),
            "章节": h.get("title"),
            "内容": (h["text"] or "")[:900],
            "相关度": h.get("score"),
        }
        if h.get("ts_code"):
            item["代码"] = h["ts_code"]
        if h.get("trade_date"):
            item["日期"] = h["trade_date"]
        if h.get("url"):
            item["链接"] = h["url"]
        out.append(item)

    res = {
        "命中": len(out),
        "可信度": conf,
        "段落": out,
        "说明": ("引用时请说明出自哪个文件/哪一节，或哪天的公告。"
                 "可信度「低」时必须告诉用户这只是弱相关内容，不能当成依据。"),
        "索引规模": st.get("块总数"),
    }
    if code_note:
        res["代码说明"] = code_note
    # 时效性护栏：知识库的公告只覆盖最近 8 天。问「最近有没有案例」时，
    # 命中几段公告会让模型以为查完了，然后回一句「知识库里没有」就收尾——
    # 那是覆盖面不够，不是真的没有。这条提示在命中非空时也要给，因为它要推的是
    # 「再搜一次」，而不是「结果不可信」。
    if _TIME_SENSITIVE.search(str(query or "")):
        res["时效提示"] = (
            "问题带时效性诉求，而本地公告只覆盖最近 8 天、方法论文档不按时间更新。"
            "如果问的是「最近有没有发生」，再用 web_search 搜一次确认，"
            "不要只凭本地库说「没有」。")
    # 规则类问题撞车护栏：问「涨跌幅限制」会命中一份回购公告并给出「高可信度」，
    # 因为"ST""涨跌幅"这些词在公告里也高频出现。不提示的话模型会把公告当规则依据，
    # 而且它看到的是"高可信度"，没有任何理由怀疑。
    try:
        from src import rules as RU
        if RU.lookup(query, limit=1).get("命中"):
            res["警告"] = ("这是市场规则/制度类问题，命中的公告或文档很可能只是关键词撞车，"
                       "不能当规则依据。规则请用 rule_lookup 查，或按通用知识直接回答。")
    except Exception:
        pass
    return res


def web_search(query: str, kind: str = "both", days: int | None = None,
               limit: int = 5) -> dict:
    """联网搜财经新闻与券商研报。本地数据和知识库都答不了时的兜底。

    实现在 src/websearch.py：只接东方财富资讯接口（0.1~0.2 秒，带日期、媒体、原文链接）。
    通用搜索引擎实测不能用于金融问答——必应搜「贵州茅台 回购 公告」返回贵州旅游攻略。

    **这不是行情数据源。** 返回的股价是新闻发布时刻的快照，可能差好几天。
    工具描述里写死了这条边界，SYSTEM 里还有一条硬性约束守着。
    """
    from src import websearch as WS  # 懒加载：不联网的机器不用背着这个依赖

    return WS.search(query, kind=kind, days=days, limit=limit)


# ---------------------------------------------------------------- 注册表
SPECS = [
    {
        "name": "market_now",
        "description": ("当前市场状态：实时指数点位与涨跌、涨跌家数、涨跌停家数、封板率、全市场成交额，"
                        "以及最近收盘日的市场档位（进攻/震荡/防守）与建议仓位上限、三因子权重。"
                        "回答任何「大盘怎么样」「现在能不能买」类问题必须先调它。"),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "portfolio_review",
        "description": ("用户持仓体检：每只票的实时价、今日涨跌、复权盈亏、仓位占比、均线位置、"
                        "系统给出的仓别建议与待处理动作（含紧急度和原因），以及组合层面的市值、集中度、档位。"
                        "回答「我的持仓怎么样」「该先处理哪只」必须先调它。"),
        "input_schema": {
            "type": "object",
            "properties": {"focus": {"type": "string", "description": "可选，只关注某只股票（名称或代码）"}},
        },
    },
    {
        "name": "stock_lookup",
        "description": ("单只股票的完整档案：实时价、今日涨跌、均线（现价口径）、20/60日位置、"
                        "区间涨跌、量比、ATR，最新财报（ROE、营收/净利同比、报告期），"
                        "以及它是否在用户持仓里、是否在最新选股榜单上及名次。"),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "股票名称或 6 位代码，如 贵州茅台 / 600519"}},
            "required": ["query"],
        },
    },
    {
        "name": "stock_history",
        "description": "单只股票近 N 个交易日的逐日收盘、涨跌幅、成交额，以及区间累计涨跌、最高最低、涨跌天数。用于回答「最近走势如何」。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "股票名称或代码"},
                "days": {"type": "integer", "description": "交易日数，默认 20，最大 60"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kline_view",
        "description": ("K 线的走势画像：区间涨跌、最大回撤、年化波动、距区间高点、"
                        "四条均线的位置与排列（多头/空头/交织）、现价是否站上 MA20/MA60、"
                        "最近一根的量能比，以及本人持仓的成本与复权盈亏。"
                        "用于回答「最近走势怎么样」「是不是放量突破」「趋势还没坏吗」"
                        "「我现在浮盈多少」。与页面的 K 线图同一份计算，结论一致。"
                        "要逐日流水用 stock_history，要形态判断用这个。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "股票名称或代码"},
                "period": {"type": "string", "enum": ["day", "week", "month"],
                           "description": "周期，默认日线。问中期趋势用 week"},
                "days": {"type": "integer", "description": "看多少根，默认 120，最大 750"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "industry_view",
        "description": ("行业视图：某只股票所属行业（或直接给行业名）今天的整体表现——"
                        "成分股数量、平均涨跌、中位数、上涨/下跌家数、成交额合计、涨幅前5与跌幅前5，"
                        "以及该股在行业内的涨跌排位。用于回答「这个板块今天怎么样」「同行业谁更强」。"),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "股票名称/代码，或行业名，如 半导体"}},
            "required": ["query"],
        },
    },
    {
        "name": "top_picks",
        "description": ("最新选股榜单前 N 名：综合分与三个分项（资金面/基本面/技术位置）、实时价、"
                        "20日涨跌、ROE、净利同比、入选理由，以及本轮被否决的股票数与原因分布。"),
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "返回前几名，默认 10"}},
        },
    },
    {
        "name": "daily_digest",
        "description": "最新日报的内容：今日要点、异动、持仓段落，以及全文（截断）。用于回答「日报说了什么」。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "data_gaps",
        "description": ("这套系统已知的数据缺口和口径限制。涉及行业板块、资金流向、消息面、"
                        "解禁质押等未接入数据的问题，必须先调它再回答，明确告诉用户「没有这个数据」。"),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "rule_lookup",
        "description": ("查本地规则卡：市场规则与制度参数本身——涨跌幅限制（含各板块与 ST）、"
                        "印花税/过户费/佣金、可转债强赎与下修条款、融资融券保证金比例、"
                        "ST 与退市标准、股东减持限制。"
                        "凡是问「规则是什么」「标准多少」「限几个点」「有什么限制」这类"
                        "关于制度的问题，先查这里；查不到再考虑联网印象作答，不要拿公告当依据。"
                        "问某只票今天涨停没有，那是行情事实，用别的工具。"),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "问题里的规则主题词，如「涨跌幅」「可转债 强赎」"}},
            "required": ["query"],
        },
    },
    {
        "name": "rag_search",
        "description": ("检索本地知识库，含四类内容：①方法论文档（复权口径怎么定、PIT 怎么判、"
                        "为什么不能用实时价算因子、哪些数据没接入、某个模块怎么实现）；"
                        "②历史报告（往期日报/选股/体检的具体内容）；③个股公告标题（最近的减持、"
                        "回购、业绩预告、问询函等，覆盖最近约 8 天）；"
                        "④公告正文关键条款（P0 类型的回购股数、募资金额、减持比例等具体数字）。"
                        "凡是问「为什么这么设计」「口径是什么」「之前怎么处理的」「有没有公告」"
                        "「公告里具体说了什么」这类问题，必须先检索再回答，不要凭印象说。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词，用具体的名词而非整个问句，如「复权因子 除权日」"},
                "kind": {"type": "string",
                         "enum": ["methodology", "report", "notice", "notice_body"],
                         "description": "限定范围：methodology=方法论文档，report=历史报告，notice=公告（标题+正文，查公告都用这个），notice_body=只要正文条款。不填则全库检索"},
                "code": {"type": "string", "description": "可选，限定某只股票（ts_code 或 6 位代码），查公告时用"},
                "days": {"type": "integer", "description": "可选，只看最近 N 天的公告/报告"},
                "k": {"type": "integer", "description": "返回几段，默认 5，最多 8"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "etf_screen",
        "description": ("ETF 筛选：按分类（宽基/行业/跨境/商品/货币/其他）、流动性、规模、"
                        "折溢价、涨跌幅筛出 ETF 列表，带现价、今日涨跌、溢价率、"
                        "近 20 日日均成交额、规模。用于回答「哪些 ETF 流动性最好」"
                        "「纳指 ETF 溢价多少」「有没有便宜的宽基」「半导体 ETF 跌了多少」。"
                        "注意：ETF 与股票是两套口径，ETF 没有 ROE/PE/净利，"
                        "查 ETF 不要用 top_picks 或 stock_lookup 的评分；"
                        "ETF 也没有涨跌幅限制。单只 ETF 的走势细节用 kline_view。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string",
                             "enum": ["宽基", "行业", "跨境", "商品", "货币", "其他"],
                             "description": "分类，不传则不限"},
                "keyword": {"type": "string",
                            "description": "名称或代码关键词，如 纳指 / 半导体 / 510300"},
                "sort": {"type": "string",
                         "enum": ["amt", "size", "premium", "discount",
                                  "ret20", "ret20_asc", "turnover"],
                         "description": ("排序列。amt=20日日均成交额（默认，流动性优先）；"
                                         "size=规模；premium=溢价高；discount=折价深；"
                                         "ret20=20日涨幅高；ret20_asc=20日跌幅大；turnover=换手")},
                "min_amount": {"type": "number",
                               "description": "20 日日均成交额下限，单位万元，默认 500"},
                "max_premium": {"type": "number",
                                "description": "溢价率上限（%），用于剔除高溢价品种，如传 1"},
                "limit": {"type": "integer", "description": "返回几条，默认 12，最多 40"},
            },
        },
    },
    {
        "name": "web_search",
        "description": ("联网搜财经新闻与券商研报（东方财富资讯库）。**只用于金融问题**，"
                        "并且只在本地工具和知识库都给不出答案时才用：监管新规、最新政策、"
                        "公司近期新闻、券商观点、某个行业最近发生了什么。"
                        "三种情况不要调用它：① 本系统已有数据的（行情、K线、持仓、选股、ETF 筛选）；"
                        "② 金融通识——PE 怎么算、定增与配股的区别、量比是什么，你自己就会；"
                        "③ 非金融问题——写代码、天气、闲聊，一律不搜，直接说这个工作台只管金融。"
                        "返回的是第三方资讯摘要，不是行情数据：里面的价格和涨跌幅是新闻发布时刻的"
                        "快照，可能已过期，涉及当前价格必须改用本系统工具。"
                        "搜不到就换个更短的关键词再搜一次（用公司简称或事件词，不要用整句问句）；"
                        "两次都空就明确说「网上也没搜到」。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "关键词，用公司简称或事件词，如「减持新规」「宁德时代 中报」「可转债 强赎」"},
                "kind": {"type": "string", "enum": ["news", "research", "both"],
                         "description": "news=财经新闻，research=券商研报，both=都要（默认）"},
                "days": {"type": "integer", "description": "可选，只要最近 N 天的内容"},
                "limit": {"type": "integer", "description": "每类返回几条，默认 5，最多 10"},
            },
            "required": ["query"],
        },
    },
]

_REGISTRY = {
    "market_now": market_now,
    "portfolio_review": portfolio_review,
    "stock_lookup": stock_lookup,
    "stock_history": stock_history,
    "kline_view": kline_view,
    "industry_view": industry_view,
    "top_picks": top_picks,
    "daily_digest": daily_digest,
    "data_gaps": data_gaps,
    "rule_lookup": rule_lookup,
    "rag_search": rag_search,
    "etf_screen": etf_screen,
    "web_search": web_search,
}

LABELS = {
    "market_now": "查市场状态",
    "portfolio_review": "查持仓体检",
    "stock_lookup": "查个股档案",
    "stock_history": "查近期走势",
    "kline_view": "看K线形态",
    "industry_view": "查行业强弱",
    "top_picks": "查选股榜单",
    "daily_digest": "读最新日报",
    "data_gaps": "核对数据缺口",
    "rule_lookup": "查规则卡",
    "rag_search": "检索知识库",
    "etf_screen": "筛ETF",
    "web_search": "联网搜索",
}


def run(name: str, args: dict | None) -> dict:
    fn = _REGISTRY.get(name)
    if not fn:
        return {"error": f"未知工具：{name}"}
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return {"error": f"参数不对：{e}"}
    except Exception as e:
        return {"error": f"执行失败：{type(e).__name__}: {str(e)[:160]}"}
