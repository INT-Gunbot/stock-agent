"""市场层：指数、广度、情绪、市场档位。

全部用 SQL 聚合，不把全市场分钟级明细拉进内存——
391 万行日线整体 load 一次要 1-2 秒，但日报每天跑一次，没必要为省这点时间
牺牲内存和可读性；真正需要全市场截面时（选股）再走批量读取。

这里算出来的市场档位（regime）是全系统共享的：选股、持仓体检、日报
都用同一个档位判断，避免同一天给出互相矛盾的结论。
"""
from __future__ import annotations

import datetime as dt

import duckdb
import pandas as pd

from . import query as Q
from .rules import limit_case_sql

INDEX_NAMES = {
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
    "000300": "沪深300",
}


def _prev_date(con, trade_date) -> dt.date | None:
    rows = con.execute(
        "SELECT max(trade_date) FROM index_bar WHERE trade_date < ?", [trade_date]
    ).fetchall()
    return rows[0][0] if rows and rows[0][0] else None


def index_overview(con, trade_date) -> list[dict]:
    """四大指数当日表现 + 均线位置。"""
    out = []
    start = pd.to_datetime(trade_date) - pd.Timedelta(days=200)
    for code, name in INDEX_NAMES.items():
        df = Q.get_index(code, start=start.date(), end=trade_date, con=con)
        if df.empty or len(df) < 2:
            continue
        px = df["close"]
        r = df.iloc[-1]
        p = df.iloc[-2]
        ma20 = px.rolling(20, min_periods=10).mean().iloc[-1]
        ma60 = px.rolling(60, min_periods=20).mean().iloc[-1]
        out.append({
            "code": code,
            "name": name,
            "close": float(r["close"]),
            "pct": (float(r["close"]) / float(p["close"]) - 1) * 100,
            "amount": float(r["amount"]) if pd.notna(r["amount"]) else None,
            "amount_chg": (
                (float(r["amount"]) / float(p["amount"]) - 1) * 100
                if pd.notna(r["amount"]) and p["amount"] else None
            ),
            "ma20": float(ma20) if pd.notna(ma20) else None,
            "ma60": float(ma60) if pd.notna(ma60) else None,
            "above_ma20": bool(pd.notna(ma20) and r["close"] > ma20),
            "above_ma60": bool(pd.notna(ma60) and r["close"] > ma60),
            "ret20": (float(r["close"]) / float(px.iloc[-21]) - 1) * 100 if len(px) > 21 else None,
        })
    return out


# 涨跌停幅度不在这里写死。原来这段 CASE 和 realtime.py 里的 _limit_pct
# 是同一规则的两份独立实现，结果 realtime 那份把注册制板块的 ST 判成了 5%，
# 盘中涨停家数和收盘涨停家数因此对不上。现在两边都从 rules.limit_case_sql 生成。
_LIMIT_CASE = limit_case_sql("b.ts_code", "d.name")

BREADTH_SQL = f"""
WITH fd AS (
    SELECT ts_code, min(trade_date) AS first_day FROM daily_bar GROUP BY 1
),
t AS (
    SELECT b.ts_code, b.trade_date, b.pct_chg, b.close, b.high, b.low, b.open, b.vol, b.amount,
           coalesce(d.name, '') AS name,
           {_LIMIT_CASE} AS lim,
           fd.first_day
    FROM daily_bar b
    LEFT JOIN dim_stock d ON d.ts_code = b.ts_code
    LEFT JOIN fd ON fd.ts_code = b.ts_code
    WHERE b.trade_date = ? AND b.vol > 0
)
SELECT
    count(*)                                                AS total,
    count(*) FILTER (WHERE pct_chg > 0)                     AS up,
    count(*) FILTER (WHERE pct_chg < 0)                     AS down,
    count(*) FILTER (WHERE pct_chg = 0)                     AS flat,
    count(*) FILTER (WHERE pct_chg >= 5)                    AS up5,
    count(*) FILTER (WHERE pct_chg <= -5)                   AS down5,
    count(*) FILTER (WHERE pct_chg >= 9.8 AND lim = 10)     AS limit_up_main,
    count(*) FILTER (WHERE pct_chg >= 19.6 AND lim = 20)    AS limit_up_gem,
    count(*) FILTER (WHERE pct_chg >= 29.4 AND lim = 30)    AS limit_up_bj,
    count(*) FILTER (WHERE pct_chg <= -9.8 AND lim = 10)    AS limit_dn_main,
    count(*) FILTER (WHERE pct_chg <= -19.6 AND lim = 20)   AS limit_dn_gem,
    count(*) FILTER (WHERE pct_chg <= -29.4 AND lim = 30)   AS limit_dn_bj,
    count(*) FILTER (WHERE pct_chg >= lim - 0.3 AND trade_date > first_day) AS touched_up,
    count(*) FILTER (WHERE pct_chg >= lim - 0.3 AND abs(high - close) < 0.005
                       AND trade_date > first_day)          AS sealed,
    median(pct_chg)                                         AS median_pct,
    avg(pct_chg)                                            AS mean_pct,
    sum(amount)                                             AS amount_total
FROM t
"""


def breadth(con, trade_date) -> dict:
    """涨跌家数、涨跌停、成交额。"""
    cur = con.execute(BREADTH_SQL, [trade_date])
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if not rows:
        return {}
    r = dict(zip(cols, rows[0]))
    total = r["total"] or 1
    limit_up = (r["limit_up_main"] or 0) + (r["limit_up_gem"] or 0) + (r["limit_up_bj"] or 0)
    limit_dn = (r["limit_dn_main"] or 0) + (r["limit_dn_gem"] or 0) + (r["limit_dn_bj"] or 0)
    r.update({
        "limit_up": limit_up,
        "limit_down": limit_dn,
        "up_ratio": (r["up"] or 0) / total * 100,
        "down_ratio": (r["down"] or 0) / total * 100,
        # 封板率 = 封住的涨停 / 触及涨停。低于 60% 说明承接弱，是退潮信号
        "seal_rate": (r["sealed"] or 0) / (r["touched_up"] or 0) * 100 if r["touched_up"] else None,
    })
    return r


def turnover_trend(con, trade_date, n: int = 20) -> dict:
    """成交额趋势：今日 vs 昨日 vs 20 日均值。

    成交额比指数更能反映"钱在不在场"。缩量上涨通常难以持续。
    """
    df = con.execute(
        """
        SELECT trade_date, sum(amount) AS amt
        FROM daily_bar WHERE trade_date <= ? AND vol > 0
        GROUP BY 1 ORDER BY 1 DESC LIMIT ?
        """,
        [trade_date, n + 1],
    ).df()
    if df.empty:
        return {}
    df = df.sort_values("trade_date")
    today = float(df["amt"].iloc[-1])
    prev = float(df["amt"].iloc[-2]) if len(df) > 1 else None
    avg20 = float(df["amt"].iloc[:-1].mean()) if len(df) > 2 else None
    # 历史分位：今日成交额在过去 60 个交易日中的位置
    hist = con.execute(
        """
        SELECT sum(amount) AS amt FROM daily_bar
        WHERE trade_date <= ? AND vol > 0
        GROUP BY trade_date ORDER BY trade_date DESC LIMIT 60
        """,
        [trade_date],
    ).df()
    pct_rank = None
    if len(hist) > 5:
        vals = hist["amt"].astype(float)
        pct_rank = float((vals < today).sum()) / len(vals) * 100
    return {
        "amount": today,
        "amount_chg": (today / prev - 1) * 100 if prev else None,
        "amount_vs_ma20": (today / avg20 - 1) * 100 if avg20 else None,
        "amount_pct_rank60": pct_rank,
    }


def regime(indices: list[dict], br: dict) -> dict:
    """市场档位：进攻 / 震荡 / 防守。

    判定顺序（先看趋势，再看广度，两者矛盾时取保守的一档）：
      进攻：主要指数在 MA60 上方 且 上涨家数占比 > 60%
      防守：主要指数跌破 MA60 且 上涨家数占比 < 35%
      其余为震荡

    这个档位决定选股权重与建议仓位上限，必须全局唯一。
    """
    sh = next((i for i in indices if i["code"] == "000001"), None)
    hs = next((i for i in indices if i["code"] == "000300"), None)
    ref = hs or sh
    up_ratio = br.get("up_ratio") or 0

    if ref is None:
        return {"level": "未知", "reason": "指数数据缺失", "position_cap": None}

    above = ref["above_ma60"]
    if above and up_ratio > 60:
        level, cap = "进攻", 80
    elif (not above) and up_ratio < 35:
        level, cap = "防守", 40
    else:
        level, cap = "震荡", 60

    reason_parts = [
        (f"{ref['name']} {'站上' if above else '跌破'} MA60 "
         f"({ref['close']:.2f} vs {ref['ma60']:.2f})") if ref.get("ma60")
        else f"{ref['name']} 数据不足",
        f"上涨家数占比 {up_ratio:.1f}%",
    ]
    return {"level": level, "reason": "，".join(reason_parts), "position_cap": cap,
            "ref_index": ref["name"]}


def build(con, trade_date) -> dict:
    prev = _prev_date(con, trade_date)
    idx = index_overview(con, trade_date)
    br = breadth(con, trade_date)
    to = turnover_trend(con, trade_date)
    return {
        "trade_date": str(trade_date)[:10],
        "prev_date": str(prev)[:10] if prev else None,
        "indices": idx,
        "breadth": br,
        "turnover": to,
        "regime": regime(idx, br),
    }
