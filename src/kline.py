"""K 线数据装配：一段可直接画图的 OHLCV 序列 + 均线 + 区间统计。

四条口径约定，每条都对应一个会画错的坑：

1. **默认前复权。** 不复权时除权日会立一根假暴跌（10 送 10 直接腰斩 50%），
   看图的人只会当成跌停。前复权以**最新交易日**为基准，所以图上的价格
   就是行情软件里的现价，视觉上和"我看到的股价"一致。
2. **盘中拼当日实时 bar。** 15:00 前库里最新一根是昨天，不拼的话图上永远缺今天。
   拼接前先比对实时昨收与库内收盘——两者不等说明今天除权，历史价要按比例重算，
   否则图上会出现一根凭空跳空的缺口（前复权的基准日变了）。
3. **均线在当前周期上算。** 周线 MA5 是"最近 5 周收盘均值"，不是把日线 MA5 抽样。
   重采样后再滚动，否则周线上的每根均线都是某一天的日线均线，毫无意义。
4. **成本线要跟着复权基准走。** positions.csv 里的 cost 是当时的真实成交价，
   前复权把历史价缩放了，直接横一条 cost 会画错位置。换算：
   cost_qfq = cost × factor(买入日) / factor(最新日)。
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

from src import query as Q
from src.db import connect

BASE = Path(__file__).resolve().parent.parent

# 周期名 -> (中文, pandas 重采样规则, 一年多少个周期)
PERIODS: dict[str, tuple[str, str | None, int]] = {
    "day": ("日线", None, 244),
    "week": ("周线", "W-FRI", 52),
    "month": ("月线", "ME", 12),
}
ADJUSTS = {"qfq": "前复权", "none": "不复权", "hfq": "后复权"}
MA_WINDOWS = (5, 10, 20, 60)


# ---------------------------------------------------------------- 代码解析
def detect_asset(code: str) -> str:
    """判断是股票还是 ETF。以 etf_dim 为准，不靠代码段猜。

    510300 走股票版的 to_ts_code 会被判成 .SZ（那段逻辑只认 60/68/00/30），
    所以这里必须查表：etf_dim 里有就是 ETF，否则按股票处理。
    """
    if not code:
        return "stock"
    con = connect(read_only=True)
    try:
        row = con.execute(
            "SELECT 1 FROM etf_dim WHERE ts_code = ? LIMIT 1", [code]
        ).fetchone()
        return "etf" if row else "stock"
    finally:
        con.close()


def resolve(q: str) -> tuple[str | None, list[dict]]:
    """代码 / 名称 -> ts_code。名称可能命中多只，一并返回候选。

    ETF 要先查 etf_dim：它的代码段（51/52/53/56/58/158/159）不在股票后缀
    推断表里，先走 dim_stock 会得到一个错的 .SZ 后缀，后面全查空。
    """
    s = str(q or "").strip().upper()
    if not s:
        return None, []
    con = connect(read_only=True)
    try:
        if s[:6].isdigit():
            code = s[:6]
            if "." in s:
                return s, []
            row = con.execute(
                "SELECT ts_code, name FROM etf_dim WHERE ts_code LIKE ? LIMIT 1",
                [code + ".%"],
            ).fetchall()
            if row:
                return row[0][0], []
            row = con.execute(
                "SELECT ts_code, name FROM dim_stock WHERE ts_code LIKE ? LIMIT 1",
                [code + ".%"],
            ).fetchall()
            if row:
                return row[0][0], []
            suffix = "SH" if code[0] in "69" else "BJ" if code[0] in "48" else "SZ"
            return f"{code}.{suffix}", []
        # 东财名称里带空格（"五 粮 液"），用户不会带着空格输入
        kw = "".join(str(q).split()).upper()
        rows = con.execute(
            "SELECT ts_code, name, 'etf' AS asset FROM etf_dim "
            "WHERE replace(replace(name, ' ', ''), '　', '') LIKE ? "
            "UNION ALL "
            "SELECT ts_code, name, 'stock' AS asset FROM dim_stock "
            "WHERE replace(replace(name, ' ', ''), '　', '') LIKE ? AND is_active "
            "ORDER BY 1 LIMIT 8",
            [f"%{kw}%", f"%{kw}%"],
        ).fetchall()
        if not rows:
            return None, []
        if len(rows) == 1:
            return rows[0][0], []
        return None, [{"ts_code": r[0], "name": r[1].replace(" ", "")} for r in rows]
    finally:
        con.close()


# ---------------------------------------------------------------- 复权与重采样
def _resample(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """把日线合成周/月线。

    用 adj_* 列聚合而不是原始价：跨除权日合成一根周线时，
    原始 open/high/low 会被不同基准的价格拼在一起（周一原始价、周五除权后价），
    算出来的影线是假的。
    """
    rule = PERIODS[period][1]
    if rule is None or df.empty:
        return df
    d = df.set_index("trade_date")
    d["_last"] = d.index          # 周期内最后一个真实交易日
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "adj_open": "first", "adj_high": "max", "adj_low": "min", "adj_close": "last",
        "vol": "sum", "amount": "sum", "_last": "last",
    }
    agg = {k: v for k, v in agg.items() if k in d.columns}
    g = d.resample(rule).agg(agg)
    g = g.dropna(subset=["close"])
    if "turnover" in d.columns:
        g["turnover"] = d["turnover"].resample(rule).sum(min_count=1)
    # 用周期内最后一个交易日的真实日期，而不是 W-FRI/月末这种"名义结束日"。
    # 否则周线最后一根的 x 轴标签会是 2026-11-06（周五），而今天是 11-03，
    # 图上出现一个未来日期，用户会以为数据穿越了。
    g["trade_date"] = g.pop("_last")
    g = g.reset_index(drop=True)
    g["pct_chg"] = (g["adj_close"] / g["adj_close"].shift(1) - 1) * 100
    return g


def _live_bar(code: str, last_date) -> dict | None:
    """取当日实时行情拼成一根未收盘的 bar。取不到返回 None。

    只有两种情况拼：①盘中/竞价；②今天已收盘但库还没更新（15:00~更新之间）。
    周末和节假日不拼——东财在非交易日仍会返回上一交易日的价格，
    照抄会造出一根周六的假 K 线。
    """
    from src import realtime as RT

    now = pd.Timestamp.now(tz=RT.CN)
    if now.weekday() >= 5:
        return None
    if last_date is not None and pd.Timestamp(last_date).date() >= now.date():
        return None  # 库里已经有今天的数据（收盘后更新过），实时价反而更粗
    st = RT.session_state()
    if st["key"] == "pre":
        return None  # 09:15 前东财把所有价格清空，没有可靠的 open/high/low
    try:
        q = (RT.quotes([code]) or {}).get(code) or {}
    except Exception:
        return None
    price = q.get("price")
    if price is None:
        return None
    open_ = q.get("open") or price
    high = q.get("high") or price
    low = q.get("low") or price
    pre = q.get("pre_close")
    return {
        "trade_date": pd.Timestamp(now.date()),
        "open": open_, "high": high, "low": low, "close": price,
        "adj_open": open_, "adj_high": high, "adj_low": low, "adj_close": price,
        "vol": q.get("vol"), "amount": q.get("amount"),
        "turnover": q.get("turnover"),
        "pct_chg": (price / pre - 1) * 100 if pre else None,
        "is_live": True,
        "session": st["text"],
        "pre_close": pre,       # 只给外层比对除权用，返回前会删掉
    }


# ---------------------------------------------------------------- 持仓
def _positions() -> dict[str, dict]:
    path = BASE / "positions.csv"
    if not path.exists():
        return {}
    try:
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    except Exception:
        return {}
    out = {}
    for r in rows:
        code = str(r.get("ts_code") or "").strip()
        if code:
            out[code] = r
    return out


def _factor_at(code: str, buy_date, asset: str = "stock") -> float | None:
    """买入日（或之前最后一个交易日）的复权因子。"""
    con = connect(read_only=True)
    table = "etf_bar" if asset == "etf" else "adj_factor"
    try:
        if buy_date:
            row = con.execute(
                f"SELECT adj_factor FROM {table} WHERE ts_code = ? AND trade_date <= ? "
                "ORDER BY trade_date DESC LIMIT 1", [code, pd.Timestamp(buy_date).date()],
            ).fetchone()
        else:
            row = con.execute(
                f"SELECT adj_factor FROM {table} WHERE ts_code = ? "
                "ORDER BY trade_date ASC LIMIT 1", [code],
            ).fetchone()
        return float(row[0]) if row and row[0] else None
    finally:
        con.close()


# ---------------------------------------------------------------- 主装配
def build(query: str, n: int = 750, period: str = "day",
          adjust: str = "qfq", with_live: bool = True,
          asset: str = "auto") -> dict:
    code, cands = resolve(query)
    if not code:
        return {"error": "没找到这只票" if not cands else "名称命中多只，请选一个",
                "candidates": cands, "query": query}
    period = period if period in PERIODS else "day"
    adjust = adjust if adjust in ADJUSTS else "qfq"
    if asset == "auto":
        asset = detect_asset(code)
    asset = "etf" if asset == "etf" else "stock"

    con = connect(read_only=True)
    try:
        if asset == "etf":
            # ETF 没有 industry，用分类（宽基/行业/跨境/商品/货币）占位，
            # 前端那一栏本来就是"行业"位，显示"跨境"比显示空更有用
            meta = con.execute(
                "SELECT name, category, TRUE FROM etf_dim WHERE ts_code = ? LIMIT 1",
                [code],
            ).fetchone()
        else:
            meta = con.execute(
                "SELECT name, industry, is_active FROM dim_stock WHERE ts_code = ? LIMIT 1",
                [code],
            ).fetchone()
        # 取全量再截断：指标要预热（MA60 需要 60 根），只取 n 根会让最左边一段没有均线
        df = Q.get_price(code, adjust=adjust, con=con, asset=asset)
    finally:
        con.close()

    if df.empty:
        return {"error": f"{code} 没有日线数据", "code": code,
                "name": meta[0] if meta else None, "query": query}

    df = df.sort_values("trade_date").reset_index(drop=True)
    # 不复权时 get_price 只返回原始价列，这里补成同名 adj_* 列，
    # 让下游（重采样、均线、统计）不必到处判分支。不复权时 adj == raw，语义上成立。
    for c in ("open", "high", "low", "close"):
        if f"adj_{c}" not in df.columns:
            df[f"adj_{c}"] = df[c]
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    if "adj_factor" not in df.columns:
        df["adj_factor"] = 1.0
    # 最新复权因子。qfq 的缩放基准、持仓盈亏、hfq 的当日 bar 都要用它，
    # 所以不分 adjust 一律算出来。
    f_now = float(df["adj_factor"].iloc[-1]) if len(df) else None

    live_note = None
    live_bar = None
    if with_live:
        bar = _live_bar(code, df["trade_date"].iloc[-1])
        if bar:
            # 今日除权：实时昨收 != 库内收盘。前复权的基准日从"库最后一天"变成"今天"，
            # 历史价必须整体乘 ratio，否则图上今天这根会凭空跳空。
            # 不复权时不能重算历史（那正是"不复权"的定义），只提示不修正。
            db_pre = float(df["close"].iloc[-1])
            pre = bar.pop("pre_close", None)
            if adjust == "hfq" and f_now:
                for c in ("adj_open", "adj_high", "adj_low", "adj_close"):
                    bar[c] = bar[c] * f_now        # 后复权：原始价 × 因子
            if pre and db_pre and abs(pre / db_pre - 1) > 0.005:
                if adjust == "qfq":
                    ratio = pre / db_pre
                    for c in ("open", "high", "low", "close",
                              "adj_open", "adj_high", "adj_low", "adj_close"):
                        df[c] = df[c] * ratio
                    live_note = f"今日除权，历史价按 {ratio:.4f} 重算"
                else:
                    live_note = "今日除权，不复权图上这根与前收的落差是除权造成的，不是下跌"
            live_bar = bar
            df = pd.concat([df, pd.DataFrame([bar])], ignore_index=True)

    df = _resample(df, period)
    # 日线也要重算涨跌幅：库里 pct_chg 在除权日按原始 pre_close 算，
    # 10 送 10 会显示 -50%，而图上的 K 线已经是复权的，两者对不上。
    df["pct_chg"] = (df["adj_close"] / df["adj_close"].shift(1) - 1) * 100
    if len(df):
        df.loc[df.index[0], "pct_chg"] = None   # 首根没有前收，不能算涨跌幅

    # 均线：在当前周期上滚动，用复权价
    px = df["adj_close"]
    for w in MA_WINDOWS:
        df[f"ma{w}"] = px.rolling(w, min_periods=max(2, w // 2)).mean()

    tail = df.tail(int(max(20, min(n, 1200)))).copy()
    t = tail["trade_date"].dt.strftime("%Y-%m-%d").tolist()

    def arr(col, d=2):
        if col not in tail.columns:
            return [None] * len(tail)
        return [None if pd.isna(v) else (round(float(v), d) if d is not None else float(v))
                for v in tail[col]]

    ma = {str(w): [None if pd.isna(v) else round(float(v), 2) for v in tail[f"ma{w}"]]
          for w in MA_WINDOWS}

    # 区间统计：用复权价算收益，用原始价算"现在多少钱"
    first_c = float(tail["adj_close"].iloc[0])
    last_c = float(tail["adj_close"].iloc[-1])
    cum = tail["adj_close"] / first_c
    per_year = PERIODS[period][2]
    rets = tail["adj_close"].pct_change().dropna()
    vol_annual = float(rets.std() * np.sqrt(per_year) * 100) if len(rets) > 2 else None
    hi, lo = float(tail["adj_high"].max()), float(tail["adj_low"].min())
    amount20 = tail["amount"].tail(20).mean() if "amount" in tail.columns else None

    stats = {
        "区间涨跌": round((last_c / first_c - 1) * 100, 2),
        "最大回撤": round(float((cum / cum.cummax() - 1).min()) * 100, 2),
        "年化波动": None if vol_annual is None else round(vol_annual, 1),
        "区间最高": round(hi, 2),
        "区间最低": round(lo, 2),
        "距高点": round((last_c / hi - 1) * 100, 2) if hi else None,
        "均额20": None if amount20 is None or pd.isna(amount20) else round(float(amount20) / 1e8, 2),
        "根数": len(tail),
        "首日": t[0], "末日": t[-1],
    }

    # 持仓成本线
    pos = _positions().get(code)
    posinfo = None
    if pos:
        try:
            cost = float(pos.get("cost") or 0)
            shares = float(pos.get("shares") or 0)
        except ValueError:
            cost = shares = 0
        if cost > 0:
            f_buy = _factor_at(code, pos.get("open_date"), asset)
            # 成本线要落在当前这张图的基准上：前复权按 factor(今日)/factor(买入日) 折算，
            # 不复权就用成交价本身（那正是"不复权"的含义）。
            line_cost = {
                "qfq": (cost * f_buy / f_now) if (f_buy and f_now) else None,
                "hfq": (cost * f_buy) if f_buy else None,
                "none": cost,
            }[adjust]
            # 盈亏不随图切换：无论看哪张图，赚了多少钱是同一个数。
            # 用复权口径算（等价于"分红再投入"），与体检页的 pnl_pct_adj 对得上。
            cq = (cost * f_buy / f_now) if (f_buy and f_now) else None
            last_raw = float(df["close"].iloc[-1])
            posinfo = {
                "shares": shares, "cost": cost,
                "cost_adj": round(line_cost, 2) if line_cost else None,
                "pnl_pct": round((last_raw / cq - 1) * 100, 2) if cq else None,
                "pnl": round((last_raw - cq) * shares, 2) if cq and shares else None,
                "open_date": (pos.get("open_date") or "")[:10],
                "tag": pos.get("tag") or "",
            }

    name = (meta[0] if meta else None) or code
    return {
        "code": code, "name": str(name).replace(" ", ""),
        "asset": asset,
        "industry": meta[1] if meta and meta[1] else None,
        "active": bool(meta[2]) if meta else None,
        "period": period, "period_label": PERIODS[period][0],
        "adjust": adjust, "adjust_label": ADJUSTS[adjust],
        "n": len(tail), "total": len(df),
        # OHLC 用复权后的列，和均线保持同一基准。用原始 close 的话，
        # 前复权模式下除权日那根 K 线会和 MA 错开一个除权缺口（茅台每年分红约 2%，
        # 一眼看不出；10 送 10 的票会直接错一倍）。
        "t": t, "o": arr("adj_open"), "h": arr("adj_high"),
        "l": arr("adj_low"), "c": arr("adj_close"),
        "raw_c": arr("close"),          # 原始收盘，给"不复权"以外的场景核对用
        "v": arr("vol", 0), "a": arr("amount", 0), "pct": arr("pct_chg"),
        "turnover": arr("turnover", 2),
        "ma": ma, "ma_windows": list(MA_WINDOWS),
        "stats": stats, "position": posinfo,
        "is_live_last": bool(tail.get("is_live", pd.Series([False])).iloc[-1])
        if "is_live" in tail.columns else False,
        "note": live_note,
        "data_last": t[-1],
    }


if __name__ == "__main__":  # 自测：python -m src.kline 600519
    import sys
    import time

    q = sys.argv[1] if len(sys.argv) > 1 else "600519"
    t0 = time.time()
    r = build(q, n=120)
    if r.get("error"):
        print("错误:", r["error"], r.get("candidates"))
        raise SystemExit(1)
    print(f"{r['name']} {r['code']}  {r['period_label']}/{r['adjust_label']}  "
          f"{r['n']}/{r['total']} 根  用时 {time.time() - t0:.2f}s")
    print("统计:", r["stats"])
    if r["position"]:
        print("持仓:", r["position"])
    for i in range(max(0, r["n"] - 4), r["n"]):
        print(f"  {r['t'][i]}  O{r['o'][i]} H{r['h'][i]} L{r['l'][i]} C{r['c'][i]}  "
              f"{r['pct'][i]}%  MA20 {r['ma']['20'][i]}")
