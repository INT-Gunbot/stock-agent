"""日报六个区块的事实组装。

设计原则（决定了这份日报有没有人看）：

1. **只报变化，不复述**。持仓 30 只票挨个点评"今天涨了/跌了"，用户三天就不看了。
   没越过阈值的票直接省略，列表里出现就意味着"有事"。
2. **所有数字在进入模板前就已经算好**，模板只负责排版。
   任何需要计算的表达式都不允许出现在渲染层，否则无法单测、无法拦截幻觉。
3. **找不到就说找不到**。公告、龙虎榜、北向资金没有数据源，就明确写"未接入"，
   让用户知道这份日报的边界在哪，而不是用似是而非的话术补空白。
4. **归因只挂可验证事件**。"主力洗盘""资金出逃"这类说法不可证伪，
   一律不写；改写"个股跑输板块多少"这种可计算的东西。

六区块固定顺序：市场概览 → 资金与情绪 → 事件与消息 → 持仓异动 → 技术面 → 明日观察。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import market as M
from .. import portfolio as P

# 异动阈值：这些数字决定"哪些票会出现在日报里"
TH_RET = 5.0        # 单日涨跌幅超过 5%
TH_VOL = 2.0        # 量能比超过 2 倍
TH_ATR = 2.0        # 单日波动超过 2 个 ATR

# 财报法定披露窗口：(起始月日, 结束月日, 报告期说明)
REPORT_WINDOWS = [
    ((4, 1), (4, 30), "年报 + 一季报"),
    ((8, 1), (8, 31), "中报"),
    ((10, 1), (10, 31), "三季报"),
]


def _pct(v, digits=2):
    return None if v is None else round(float(v), digits)


def _fmt_pct(v, digits=2) -> str:
    """原始百分比数值 -> "12.34%"。源数据可能给到 10 位小数，直接插模板会很难看。"""
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):+.{digits}f}%"


def _amt(v):
    """成交额转"亿元"字符串。"""
    if v is None or pd.isna(v):
        return None
    return round(float(v) / 1e8, 0)


# ---------------------------------------------------------------- 区块 1
def block_market(mk: dict) -> dict:
    br = mk.get("breadth", {})
    return {
        "trade_date": mk["trade_date"],
        "prev_date": mk["prev_date"],
        "indices": [
            {
                "name": i["name"],
                "close": round(i["close"], 2),
                "pct": _pct(i["pct"]),
                "amount_yi": _amt(i["amount"]),
                "amount_chg": _pct(i["amount_chg"], 1),
                "above_ma20": i["above_ma20"],
                "above_ma60": i["above_ma60"],
                "ret20": _pct(i.get("ret20"), 1),
            }
            for i in mk.get("indices", [])
        ],
        "breadth": {
            "up": br.get("up"), "down": br.get("down"), "flat": br.get("flat"),
            "total": br.get("total"),
            "up_ratio": _pct(br.get("up_ratio"), 1),
            "limit_up": br.get("limit_up"), "limit_down": br.get("limit_down"),
            "median_pct": _pct(br.get("median_pct")),
        },
        "regime": mk.get("regime", {}),
    }


# ---------------------------------------------------------------- 区块 2
def sentiment_score(br: dict, to: dict) -> dict:
    """情绪分（0-100），四个可验证的分项加权。

    这个分数不进任何交易决策，只是把"今天市场好还是差"压成一个可比数字，
    方便回看时快速定位（比如"情绪分连续 5 天低于 30"比翻 5 份日报容易）。
    """
    up_ratio = br.get("up_ratio") or 50
    limit_up = br.get("limit_up") or 0
    limit_dn = br.get("limit_down") or 0
    seal = br.get("seal_rate")

    parts = {
        "breadth": (up_ratio - 50) * 0.6,                       # ±30
        "limit_up": min(limit_up / 50, 1.0) * 10,               # 0~10
        "limit_down": -min(limit_dn / 20, 1.0) * 10,            # -10~0
        "seal": ((seal - 70) / 30 * 10) if seal is not None else 0,  # ±10
    }
    score = 50 + sum(parts.values())
    score = max(0.0, min(100.0, score))
    label = ("过热" if score >= 80 else "偏强" if score >= 60
             else "中性" if score >= 40 else "偏弱" if score >= 20 else "冰点")
    return {"score": round(score, 0), "label": label,
            "parts": {k: round(v, 1) for k, v in parts.items()}}


def block_sentiment(mk: dict) -> dict:
    br = mk.get("breadth", {})
    to = mk.get("turnover", {})
    return {
        "amount_yi": _amt(to.get("amount")),
        "amount_chg": _pct(to.get("amount_chg"), 1),
        "amount_vs_ma20": _pct(to.get("amount_vs_ma20"), 1),
        "amount_pct_rank60": _pct(to.get("amount_pct_rank60"), 0),
        "up5": br.get("up5"), "down5": br.get("down5"),
        "limit_up": br.get("limit_up"), "limit_down": br.get("limit_down"),
        "touched_up": br.get("touched_up"),
        "seal_rate": _pct(br.get("seal_rate"), 0),
        "score": sentiment_score(br, to),
        # 诚实声明：这些指标现在拿不到，写清楚比编一个近似值强
        "missing": [
            "北向资金净流入（免费源已停更沪深股通实时额度）",
            "两融余额（T+1 披露，且需单独接口）",
            "主力资金净流入（东财口径需按票单拉，全市场约 5000 次请求，暂不纳入日报）",
            "龙虎榜（盘后披露，暂未接入）",
        ],
    }


# ---------------------------------------------------------------- 区块 3
def block_events(con, trade_date, hold: pd.DataFrame) -> dict:
    """事件与消息面：只用现有数据能推出的可验证事件。

    这里刻意不做"新闻摘要"——没有公告接口就承认没有，
    用行情数据硬猜消息面只会产出错误归因。
    """
    td = pd.to_datetime(trade_date)
    events = []

    # 1) 全市场当日除权除息家数 + 明细
    # 只认因子"上升且幅度 > 0.5%"：分红送转一定让复权因子上升（缩股极少见），
    # 而 0.5% 以下的变化多半来自兜底因子的日常抖动（北交所有 358 只票
    # 新浪不覆盖，用东财后复权价反推，那条序列本身每天漂移约 0.2%）。
    exdiv = con.execute(
        """
        WITH x AS (
            SELECT ts_code, trade_date, adj_factor,
                   lag(adj_factor) OVER (PARTITION BY ts_code ORDER BY trade_date) AS prev_f
            FROM adj_factor WHERE trade_date BETWEEN ? AND ?
        )
        SELECT x.ts_code, coalesce(d.name,'') AS name, x.prev_f, x.adj_factor
        FROM x LEFT JOIN dim_stock d ON d.ts_code = x.ts_code
        WHERE x.trade_date = ? AND x.prev_f IS NOT NULL
          AND x.adj_factor / x.prev_f - 1 > 0.005
        ORDER BY x.adj_factor / x.prev_f DESC
        """,
        [td - pd.Timedelta(days=10), td, td.date()],
    ).df()
    if len(exdiv):
        events.append({
            "type": "除权除息",
            "level": "info",
            "title": f"今日 {len(exdiv)} 只个股除权除息",
            "detail": "除权日原始涨跌幅会失真（10 送 10 显示 -50%），"
                      "本日报的涨跌幅已按复权口径修正。",
            "items": [
                {"name": r["name"] or r["ts_code"],
                 "value": f"复权因子 {r['prev_f']:.4f}→{r['adj_factor']:.4f} "
                          f"({(r['adj_factor']/r['prev_f']-1)*100:+.2f}%)"}
                for _, r in exdiv.head(8).iterrows()
            ],
        })

    # 2) 持仓股近 10 日是否有新财报可见
    fin_events = []
    for _, h in hold.iterrows():
        if h.get("error"):
            continue
        r = con.execute(
            """
            SELECT end_date, visible_date, net_profit_yoy, revenue_yoy
            FROM fina_yjbb WHERE ts_code = ? ORDER BY visible_date DESC LIMIT 1
            """,
            [h["ts_code"]],
        ).fetchall()
        if not r:
            continue
        end_d, vis_d, npy, rvy = r[0]
        if vis_d and (td.date() - vis_d).days <= 10:
            fin_events.append({
                "name": h["name"],
                "value": f"报告期 {end_d}，净利同比 {_fmt_pct(npy)}"
                         f"，营收同比 {_fmt_pct(rvy)}",
            })
    if fin_events:
        events.append({
            "type": "财报披露", "level": "info",
            "title": f"持仓股近 10 日内有 {len(fin_events)} 条财报更新",
            "detail": "数据来自东财业绩报表，可见日按法定披露截止日保守推算。",
            "items": fin_events,
        })

    # 3) 披露窗口提示：按日期判断当前是否处在密集披露期
    md = (td.month, td.day)
    in_window = None
    for (s, e_val, desc) in REPORT_WINDOWS:
        if s <= md <= e_val:
            in_window = desc
            break
    if in_window:
        events.append({
            "type": "披露窗口", "level": "warn",
            "title": f"当前处于 {in_window} 密集披露期",
            "detail": "业绩预告/快报集中发布，个股可能因业绩缺口出现单日大幅波动，"
                      "持仓票建议核对是否已披露。",
            "items": [],
        })

    # 4) 持仓股异常：连续下跌 3 日以上
    streak = []
    for _, h in hold.iterrows():
        if h.get("error"):
            continue
        rows = con.execute(
            """
            SELECT trade_date, close, pre_close FROM daily_bar
            WHERE ts_code = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 6
            """,
            [h["ts_code"], td.date()],
        ).df()
        if len(rows) < 4:
            continue
        n = 0
        for _, rr in rows.iloc[1:].iterrows():
            if rr["close"] < rr["pre_close"]:
                n += 1
            else:
                break
        if n >= 3:
            streak.append({"name": h["name"], "value": f"连续 {n} 个交易日下跌"})
    if streak:
        events.append({
            "type": "连续下跌", "level": "warn",
            "title": f"{len(streak)} 只持仓连续下跌 3 日以上",
            "detail": "连续下跌本身不构成卖出理由，但值得核对是否有未注意到的信息。",
            "items": streak,
        })

    gaps = [
        "公司公告全文（需接入公告接口，目前只能靠人工核对交易所网站）",
        "研报观点与评级变动",
        "行业政策与新闻（无行业分类字段，无法自动关联）",
    ]
    return {"events": events, "gaps": gaps}


# ---------------------------------------------------------------- 区块 4
def block_holdings(hold: pd.DataFrame, mk: dict) -> dict:
    if hold.empty:
        return {"summary": {}, "alerts": [], "quiet": [], "all": []}
    s = P.summary(hold)
    bench = next((i for i in mk.get("indices", []) if i["code"] == "000300"), None)
    bench_ret = bench["pct"] if bench else None

    alerts, quiet = [], []
    for _, h in hold.iterrows():
        if h.get("error"):
            alerts.append({
                "name": h["name"], "ts_code": h["ts_code"], "reasons": [h["error"]],
                "day_ret": None, "pnl_pct": None, "weight": None, "severity": 3,
            })
            continue
        reasons, sev = [], 0

        if h["day_ret"] is not None and abs(h["day_ret"]) >= TH_RET:
            reasons.append(f"单日 {h['day_ret']:+.2f}%，超过 ±{TH_RET:.0f}% 阈值")
            sev = max(sev, 2)
        atr_pct = h.get("atr_pct")
        if (h["day_ret"] is not None and atr_pct and atr_pct > 0
                and abs(h["day_ret"]) / atr_pct >= TH_ATR):
            reasons.append(f"波动达 {abs(h['day_ret'])/atr_pct:.1f} 倍 ATR"
                           f"（该股日常波动 {atr_pct:.1f}%）")
            sev = max(sev, 2)
        vr = h.get("vol_ratio")
        if vr and vr >= TH_VOL:
            reasons.append(f"量能放大至 {vr:.1f} 倍（20 日均量基准）")
            sev = max(sev, 2)
        if vr and vr <= 0.5:
            reasons.append(f"量能萎缩至 {vr:.2f} 倍，缩量明显")
            sev = max(sev, 1)

        # 均线：只报"穿越"这个动作，不报"在下方"这个状态。
        # 一只票跌破 MA20 后可能趴两个月，天天列出来会把真正的新变化淹掉。
        if not h["above_ma20"] and h.get("ma20"):
            if h.get("prev_above_ma20"):
                reasons.append(f"今日跌破 MA20（{h['close']:.2f} vs {h['ma20']:.2f}），昨日还在上方")
                sev = max(sev, 2)
        elif h["above_ma20"] and h.get("prev_above_ma20") is False:
            reasons.append(f"今日收复 MA20（{h['close']:.2f} vs {h['ma20']:.2f}）")
            sev = max(sev, 1)
        elif h["above_ma20"] and h.get("ret5") is not None and h["ret5"] < -3:
            reasons.append(f"虽在 MA20 上方，但 5 日已跌 {h['ret5']:.1f}%")
            sev = max(sev, 1)

        # 位置：只在创 20 日新高/新低时提，中间位置没有信息量
        pos20 = h.get("pos20")
        if pos20 is not None and pos20 >= 0.999:
            reasons.append("创 20 日新高")
            sev = max(sev, 1)
        if pos20 is not None and pos20 <= 0.001:
            reasons.append("创 20 日新低")
            sev = max(sev, 2)

        if h.get("is_exdiv"):
            reasons.append("今日除权，成本价需核对是否已由券商摊薄")
            sev = max(sev, 2)

        item = {
            "name": h["name"], "ts_code": h["ts_code"], "tag": h.get("tag", ""),
            "close": round(float(h["close"]), 2),
            "day_ret": _pct(h["day_ret"]),
            "pnl_pct": _pct(h["pnl_pct"]),
            "pnl": round(float(h["pnl"]), 0),
            "weight": _pct(h["weight"], 1),
            "day_pnl": round(float(h["day_pnl"]), 0) if h.get("day_pnl") is not None else None,
            "vol_ratio": _pct(vr, 2),
            "ma20": _pct(h.get("ma20")),
            "ma60": _pct(h.get("ma60")),
            "pos20": _pct(pos20, 2),
            "ret20": _pct(h.get("ret20"), 1),
            "reasons": reasons,
            "severity": sev,
            # 超额收益：剥离市场 beta 后才能谈个股自身逻辑
            "excess": _pct(h["day_ret"] - bench_ret, 2)
            if (bench_ret is not None and h["day_ret"] is not None) else None,
        }
        (alerts if reasons else quiet).append(item)

    alerts.sort(key=lambda x: (-x["severity"], -(abs(x["day_ret"] or 0))))

    # 完整持仓表：异动是重点，但"今天全都没事"本身也是信息，
    # 用户需要一眼看到全部持仓的位置，而不是只看到被过滤后的几只
    table = []
    for _, h in hold.iterrows():
        if h.get("error"):
            continue
        table.append({
            "name": h["name"], "tag": h.get("tag", ""),
            "close": round(float(h["close"]), 2),
            "day_ret": _pct(h["day_ret"]),
            "weight": _pct(h["weight"], 1),
            "pnl_pct": _pct(h["pnl_pct"]),
            "pos20": _pct(h.get("pos20"), 2),
            "ma20_side": "上方" if h["above_ma20"] else "下方",
            "ma60_side": "上方" if h["above_ma60"] else "下方",
            "ret20": _pct(h.get("ret20"), 1),
        })
    table.sort(key=lambda x: -(x["weight"] or 0))

    return {
        "summary": {
            "mv": round(s["mv"], 0),
            "cost": round(s["cost"], 0),
            "pnl": round(s["pnl"], 0),
            "pnl_pct": _pct(s["pnl_pct"]),
            "day_pnl": round(s["day_pnl"], 0),
            "day_ret": _pct(s["day_ret"]),
            "excess": _pct((s["day_ret"] - bench_ret), 2)
            if (bench_ret is not None and s.get("day_ret") is not None) else None,
            "bench_name": bench["name"] if bench else None,
            "bench_ret": _pct(bench_ret),
            "positions": s["positions"],
            "winners": s["winners"], "losers": s["losers"],
            "above_ma20": s["above_ma20"], "above_ma60": s["above_ma60"],
            "top3_weight": _pct(s["top3_weight"], 1),
            "max_weight": _pct(s["max_weight"], 1),
            "max_weight_name": s["max_weight_name"],
        },
        "alerts": alerts,
        "quiet": [{"name": x["name"], "day_ret": x["day_ret"]} for x in quiet],
        "all": alerts + quiet,
        "table": table,
    }


# ---------------------------------------------------------------- 区块 5
def block_technical(mk: dict, hold: pd.DataFrame, con=None, trade_date=None) -> dict:
    """技术面观察：指数与持仓的均线结构 + 全市场高低点分布。"""
    idx_rows = []
    for i in mk.get("indices", []):
        structure = ("多头排列" if i["above_ma20"] and i["above_ma60"]
                     else "破位" if (not i["above_ma20"] and not i["above_ma60"])
                     else "短期弱/中期强" if not i["above_ma20"] else "短期强/中期弱")
        idx_rows.append({
            "name": i["name"], "close": round(i["close"], 2), "pct": _pct(i["pct"]),
            "ma20": _pct(i.get("ma20")), "ma60": _pct(i.get("ma60")),
            "above_ma20": i["above_ma20"], "above_ma60": i["above_ma60"],
            "structure": structure, "ret20": _pct(i.get("ret20"), 1),
        })

    hold_rows = []
    for _, h in hold.iterrows():
        if h.get("error"):
            continue
        hold_rows.append({
            "name": h["name"], "close": round(float(h["close"]), 2),
            "ma20": _pct(h.get("ma20")), "ma60": _pct(h.get("ma60")),
            "above_ma20": h["above_ma20"], "above_ma60": h["above_ma60"],
            "pos20": _pct(h.get("pos20"), 2), "pos60": _pct(h.get("pos60"), 2),
            "atr_pct": _pct(h.get("atr_pct"), 1),
            "ret20": _pct(h.get("ret20"), 1), "ret60": _pct(h.get("ret60"), 1),
            "hi20": _pct(h.get("hi20")), "lo20": _pct(h.get("lo20")),
        })

    market_hl = {}
    if con is not None and trade_date is not None:
        try:
            rows = con.execute(
                """
                WITH w AS (
                    SELECT ts_code, trade_date, close,
                           max(close) OVER (PARTITION BY ts_code ORDER BY trade_date
                                            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS hi60,
                           min(close) OVER (PARTITION BY ts_code ORDER BY trade_date
                                            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS lo60,
                           row_number() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) AS rn
                    FROM daily_bar WHERE trade_date >= ?
                )
                SELECT count(*) FILTER (WHERE close >= hi60) AS new_high,
                       count(*) FILTER (WHERE close <= lo60) AS new_low,
                       count(*) AS n
                FROM w WHERE rn = 1
                """,
                [pd.to_datetime(trade_date) - pd.Timedelta(days=100)],
            ).fetchall()
            if rows:
                nh, nl, n = rows[0]
                market_hl = {"new_high60": nh, "new_low60": nl, "sample": n,
                             "high_ratio": round(nh / n * 100, 1) if n else None}
        except Exception:
            market_hl = {}

    return {"indices": idx_rows, "holdings": hold_rows, "market": market_hl}


# ---------------------------------------------------------------- 区块 6
def block_watch(mk: dict, hold: pd.DataFrame) -> list[dict]:
    """明日观察清单：只给条件框架，不给涨跌预测。

    每条格式固定为"若 A 则 B"，A 必须是可观测的价格/事件，
    B 必须是可执行的动作或明确的观察结论。
    """
    out = []
    hs = next((i for i in mk.get("indices", []) if i["code"] == "000300"), None)
    sh = next((i for i in mk.get("indices", []) if i["code"] == "000001"), None)

    for idx, label in ((hs, "沪深300"), (sh, "上证指数")):
        if not idx:
            continue
        if idx.get("ma20"):
            direction = "守住" if idx["above_ma20"] else "收复"
            out.append({
                "scope": "市场",
                "cond": f"{label}{direction} MA20（{idx['ma20']:.2f}）",
                "then": "市场档位维持当前判断；若进一步站上 MA60，进攻档参数才生效",
                "price": round(idx["ma20"], 2),
                "kind": "index",
            })
        if idx.get("ma60"):
            out.append({
                "scope": "市场",
                "cond": f"{label}跌破 MA60（{idx['ma60']:.2f}）",
                "then": "按防守档处理：选股偏基本面、仓位上限下调至 40%、持仓审核更严",
                "price": round(idx["ma60"], 2),
                "kind": "index",
            })

    for _, h in hold.iterrows():
        if h.get("error"):
            continue
        # 每只票最多 2 条，否则 6 只持仓就能撑满整张表，
        # 用户第二天就不会再往下看了
        added = 0
        if h.get("ma20") and h.get("close"):
            gap = (h["ma20"] / h["close"] - 1) * 100
            # 只保留现价 12% 以内的关键位：再远的价位一周内都碰不到，
            # 列出来只会稀释真正需要盯的那几条
            if abs(gap) <= 12:
                if h["above_ma20"]:
                    cond = f"跌破 MA20（{h['ma20']:.2f}，下方 {abs(gap):.1f}% 处）"
                    then = ("视为短期趋势走坏：观察仓先减半，核心仓结合基本面再判断"
                            if h.get("tag") != "核心仓"
                            else "核心仓先不动，但要核对当初的持有逻辑是否还成立")
                else:
                    cond = f"收复 MA20（{h['ma20']:.2f}，需上涨 {abs(gap):.1f}%）"
                    then = "收复前不补仓；站上并守住再考虑，避免在下跌中继里加仓"
                out.append({"scope": h["name"], "cond": cond, "then": then,
                            "price": round(h["ma20"], 2), "kind": "ma20"})
                added += 1
        if added < 2 and h.get("lo20") and h.get("close"):
            gap_lo = (h["lo20"] / h["close"] - 1) * 100
            if abs(gap_lo) <= 12:
                out.append({
                    "scope": h["name"],
                    "cond": f"跌破 20 日低点（{h['lo20']:.2f}，下方 {abs(gap_lo):.1f}%）",
                    "then": "确认破位，按纪律止损，不再以'再等一天'为由拖延",
                    "price": round(h["lo20"], 2), "kind": "lo20",
                })
                added += 1

    return out


def build_summary(mk: dict, se: dict, ho: dict) -> dict:
    """今日要点：3 句话，让用户 30 秒知道发生了什么。

    全部由规则模板拼接，不接 LLM。理由：这是全篇唯一"总结性"文字，
    一旦让模型自己描述，它很容易把"指数涨 1% 但都在均线下方"说成
    "市场转强"，而这句话恰恰会决定用户怎么读后面所有内容。
    """
    idx = mk.get("indices", [])
    b = mk.get("breadth", {})
    lines = []

    if idx:
        ups = [i for i in idx if i["pct"] > 0]
        best = max(idx, key=lambda x: x["pct"])
        worst = min(idx, key=lambda x: x["pct"])
        if len(ups) == len(idx):
            head = "四大指数全线上涨"
        elif not ups:
            head = "四大指数全线下跌"
        else:
            head = f"{len(ups)}/{len(idx)} 个指数上涨"
        lines.append(
            f"{head}，{best['name']}领涨 {best['pct']:+.2f}%、"
            f"{worst['name']} {worst['pct']:+.2f}%；"
            f"涨跌家数 {b.get('up')}:{b.get('down')}（上涨占比 "
            f"{(b.get('up_ratio') or 0):.0f}%），涨停 {b.get('limit_up')} 家、"
            f"跌停 {b.get('limit_down')} 家。"
        )

    above20 = sum(1 for i in idx if i["above_ma20"])
    above60 = sum(1 for i in idx if i["above_ma60"])
    if idx:
        if above20 == 0 and above60 == 0:
            lines.append(
                f"但 {len(idx)} 个指数全部位于 MA20 与 MA60 下方，"
                "今天的上涨仍属下跌趋势中的反弹，不宜按进攻档操作。"
            )
        elif above20 == len(idx) and above60 == len(idx):
            lines.append(f"{len(idx)} 个指数全部站上 MA20 与 MA60，中期趋势结构完好。")
        else:
            lines.append(f"指数位置分化：{above20}/{len(idx)} 在 MA20 上方，"
                         f"{above60}/{len(idx)} 在 MA60 上方。")

    s = ho.get("summary") or {}
    if s:
        tone = "跑赢" if (s.get("excess") or 0) > 0 else "跑输"
        lines.append(
            f"持仓当日 {s['day_ret']:+.2f}%，{tone}沪深300 "
            f"{abs(s['excess'] or 0):.2f}pct；"
            f"{s['positions']} 只中 {s['above_ma20']} 只在 MA20 上方。"
        )
        alerts = ho.get("alerts") or []
        if alerts:
            names = "、".join(a["name"] for a in alerts[:3])
            lines.append(f"需要留意：{names}{'等' if len(alerts) > 3 else ''}触发异动阈值。")

    return {"lines": lines, "score": se.get("score", {}).get("label"),
            "regime": mk.get("regime", {}).get("level")}


# ---------------------------------------------------------------- 组装
def build(con, trade_date=None, positions: pd.DataFrame | None = None) -> dict:
    if trade_date is None:
        rows = con.execute("SELECT max(trade_date) FROM daily_bar").fetchall()
        trade_date = rows[0][0]
    td = pd.to_datetime(trade_date).date()

    mk = M.build(con, td)
    hold = P.enrich(con, td, positions)

    blocks = {
        "market": block_market(mk),
        "sentiment": block_sentiment(mk),
        "events": block_events(con, td, hold),
        "holdings": block_holdings(hold, mk),
        "technical": block_technical(mk, hold, con, td),
        "watch": block_watch(mk, hold),
    }
    blocks["summary"] = build_summary(mk, blocks["sentiment"], blocks["holdings"])

    gaps = []
    if not hold.empty and len(hold) and hold["error"].notna().any():
        gaps.append("部分持仓无行情数据：" + "、".join(
            hold[hold["error"].notna()]["name"].tolist()))

    return {
        "meta": {
            "trade_date": str(td),
            "prev_date": mk.get("prev_date"),
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_gaps": gaps,
            "thresholds": {"ret": TH_RET, "vol_ratio": TH_VOL, "atr": TH_ATR},
        },
        **blocks,
    }
