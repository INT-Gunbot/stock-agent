"""统一取数入口。

上层（因子引擎、日报、持仓体检）只许通过本模块取数，禁止直接查表——
这样复权口径、point-in-time 口径改一次就全局生效。
"""
from __future__ import annotations

import datetime as dt

import duckdb
import pandas as pd

from .db import connect


def _as_date(v) -> dt.date | None:
    if v is None:
        return None
    if isinstance(v, dt.date):
        return v
    return pd.to_datetime(v).date()


def attach_return(df: pd.DataFrame, exdiv_gap: float = 0.002) -> pd.DataFrame:
    """为日线附加真实涨跌幅列 `ret`。

    规则：
    - 非除权日：`ret = close / pre_close - 1`，与行情软件显示一致
    - 除权日：原始涨跌幅因除权跳空失真（10 送 10 显示 -50%），
      改用复权收益率 `adj_close / adj_close.shift(1) - 1`

    除权判定：复权因子单日变化超过 exdiv_gap。

    为什么现在可以用这么简单的判据：
    因子已改为新浪的阶梯因子，只在除权日跳变，非除权日恒定。
    早期用"东财后复权价 ÷ 原始价"反推的因子每天漂移 ±2%，
    与当日涨跌幅相关性达 -0.91，任何基于它的除权判定都会误判——
    那个方案已整体废弃，见 update.update_factors 的注释。
    """
    if df.empty:
        return df
    out = df.copy()
    raw_ret = out["close"] / out["pre_close"] - 1
    adj_ret = out["adj_close"] / out["adj_close"].shift(1) - 1
    if "adj_factor" in out.columns:
        factor_chg = (out["adj_factor"] / out["adj_factor"].shift(1) - 1).abs()
        is_exdiv = factor_chg > exdiv_gap
    else:
        is_exdiv = (adj_ret - raw_ret).abs() > 0.03
    out["ret"] = adj_ret.where(is_exdiv, raw_ret)
    out["is_exdiv"] = is_exdiv.fillna(False)
    return out


def get_stock_list(con: duckdb.DuckDBPyConnection | None = None,
                   active_only: bool = True) -> pd.DataFrame:
    own = con is None
    con = con or connect(read_only=True)
    try:
        sql = "SELECT * FROM dim_stock"
        if active_only:
            sql += " WHERE is_active"
        return con.execute(sql + " ORDER BY ts_code").df()
    finally:
        if own:
            con.close()


def get_price(ts_code: str,
              start=None,
              end=None,
              adjust: str = "hfq",
              con: duckdb.DuckDBPyConnection | None = None,
              asset: str = "stock") -> pd.DataFrame:
    """取日线并动态复权。

    adjust: "none" 原始价 / "hfq" 后复权 / "qfq" 前复权（以区间最后一天为基准）
    asset:  "stock" 走 daily_bar + adj_factor；"etf" 走 etf_bar（因子同表）
    返回列含 close(原始) 与 adj_close(复权后)，技术指标一律用 adj_close。

    两条链路的复权公式完全一致，因为 etf_bar.adj_factor 已经归一成
    "首日=1、越大越新"的股票口径。ETF 因子若直接存后向系数（末期=1），
    这里的前复权会把历史价反向放大，图和收益全反。
    """
    own = con is None
    con = con or connect(read_only=True)
    try:
        if asset == "etf":
            sql = """
                SELECT b.ts_code, b.trade_date, b.open, b.high, b.low, b.close,
                       b.pre_close, b.vol, b.amount, b.pct_chg, b.turnover,
                       b.adj_factor
                FROM etf_bar b
                WHERE b.ts_code = ?
            """
        else:
            sql = """
                SELECT b.ts_code, b.trade_date, b.open, b.high, b.low, b.close,
                       b.pre_close, b.vol, b.amount, b.pct_chg, b.turnover,
                       a.adj_factor
                FROM daily_bar b
                LEFT JOIN adj_factor a
                       ON a.ts_code = b.ts_code AND a.trade_date = b.trade_date
                WHERE b.ts_code = ?
            """
        params: list = [ts_code]
        if start is not None:
            sql += " AND b.trade_date >= ?"
            params.append(_as_date(start))
        if end is not None:
            sql += " AND b.trade_date <= ?"
            params.append(_as_date(end))
        sql += " ORDER BY b.trade_date"
        df = con.execute(sql, params).df()
    finally:
        if own:
            con.close()

    if df.empty:
        return df

    # 统一时间列为 datetime64，避免与上游 date 对象 merge 时报类型冲突
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["adj_factor"] = df["adj_factor"].ffill().bfill()
    if adjust == "hfq":
        df["adj_close"] = df["close"] * df["adj_factor"]
        for c in ("open", "high", "low"):
            df[f"adj_{c}"] = df[c] * df["adj_factor"]
    elif adjust == "qfq":
        base = df["adj_factor"].iloc[-1]
        df["adj_close"] = df["close"] * df["adj_factor"] / base
        for c in ("open", "high", "low"):
            df[f"adj_{c}"] = df[c] * df["adj_factor"] / base
    else:
        df["adj_close"] = df["close"]
    return df


def get_fina(ts_code: str,
             as_of=None,
             con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Point-in-time 财务数据：只返回 as_of 当日已公告的记录。

    as_of 为空则取全部（含未来），仅用于研究，不可用于回测。
    同一报告期若有多个版本，取最新入库的那一版（正确反映追溯调整后的数字）。
    """
    own = con is None
    con = con or connect(read_only=True)
    try:
        base = """
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY ts_code, end_date ORDER BY ingested_at DESC
                ) AS rn
                FROM fina_yjbb
                WHERE ts_code = ?
        """
        params: list = [ts_code]
        if as_of is not None:
            base += " AND visible_date <= ?"
            params.append(_as_date(as_of))
        base += """
            )
            SELECT * FROM ranked WHERE rn = 1 ORDER BY end_date DESC
        """
        df = con.execute(base, params).df()
        if not df.empty:
            df["end_date"] = pd.to_datetime(df["end_date"])
            df["ann_date"] = pd.to_datetime(df["ann_date"])
            df["visible_date"] = pd.to_datetime(df["visible_date"])
        return df
    finally:
        if own:
            con.close()


def get_fina_latest(ts_code: str,
                    as_of=None,
                    con: duckdb.DuckDBPyConnection | None = None) -> dict | None:
    """取最近一期财务记录，返回 dict；无数据返回 None。"""
    df = get_fina(ts_code, as_of=as_of, con=con)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_index(index_code: str, start=None, end=None,
              con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    own = con is None
    con = con or connect(read_only=True)
    try:
        sql = "SELECT * FROM index_bar WHERE index_code = ?"
        params: list = [index_code]
        if start is not None:
            sql += " AND trade_date >= ?"
            params.append(_as_date(start))
        if end is not None:
            sql += " AND trade_date <= ?"
            params.append(_as_date(end))
        df = con.execute(sql + " ORDER BY trade_date", params).df()
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df
    finally:
        if own:
            con.close()


def get_trade_dates(con=None) -> list:
    """由日线数据反推交易日历。"""
    own = con is None
    con = con or connect(read_only=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT trade_date FROM daily_bar ORDER BY trade_date"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        if own:
            con.close()


def latest_trade_date(con=None) -> dt.date | None:
    own = con is None
    con = con or connect(read_only=True)
    try:
        rows = con.execute("SELECT max(trade_date) FROM daily_bar").fetchall()
        return rows[0][0] if rows else None
    finally:
        if own:
            con.close()
