"""AkShare 数据源适配层（免费源）。

所有接口统一返回英文列名 + 标准 ts_code，屏蔽上游字段变动。
换数据源（Tushare / 其他）时只需重写本文件，上层不受影响。

关于复权（重要）：
东财不复权接口返回的 pre_close 就是"上一交易日收盘"，并不做除权调整，
其 pct_chg 在除权日是按未调整的前收计算的 —— 除权日的涨跌幅是错的。
因此这里改为同时拉取后复权价，用 hfq_close / close 反推每日复权因子，
保证 adj_factor 在除权日准确跳变。代价是每只票 2 次请求。
"""
from __future__ import annotations

import datetime as dt
import time

import akshare as ak
import pandas as pd

# 东财日线中文列 -> 英文列
_HIST_MAP = {
    "日期": "trade_date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "vol",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "price_chg",
    "换手率": "turnover",
}

_INDEX_MAP = {
    "日期": "trade_date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "vol",
    "成交额": "amount",
}

BAR_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "vol", "amount", "amplitude", "pct_chg", "price_chg", "turnover"]


def _retry(fn, *args, retries: int = 3, wait: float = 1.0, **kwargs):
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # 网络抖动 / 反爬 / 接口变更
            last = e
            time.sleep(wait * (i + 1))
    raise RuntimeError(f"{getattr(fn, '__name__', fn)} 连续 {retries} 次失败: {last}")


def to_ts_code(symbol: str) -> str:
    """6 位代码 -> 标准 ts_code。"""
    s = str(symbol).zfill(6)
    if s.startswith(("60", "68", "90")):
        return f"{s}.SH"
    if s.startswith(("00", "30", "20")):
        return f"{s}.SZ"
    if s.startswith(("43", "83", "87", "88", "92")):
        return f"{s}.BJ"
    return f"{s}.SZ"


def to_symbol(ts_code: str) -> str:
    return ts_code.split(".")[0]


def fetch_stock_list() -> pd.DataFrame:
    """全 A 股列表。

    优先用东财实时快照（单次请求拿全市场，稳定），
    失败时降级到新浪代码表（该接口会访问 bse.cn，网络环境不佳时可能不通）。
    """
    try:
        raw = ak.stock_zh_a_spot_em()
        raw = raw.rename(columns={"代码": "symbol", "名称": "name"})[["symbol", "name"]]
    except Exception:
        raw = _retry(ak.stock_info_a_code_name).rename(columns={"code": "symbol"})
        raw = raw[["symbol", "name"]]

    df = raw.copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["ts_code"] = df["symbol"].map(to_ts_code)
    df["exchange"] = df["ts_code"].str.split(".").str[1]
    df["industry"] = None
    df["is_active"] = True
    df["updated_at"] = pd.Timestamp.now()
    return df[["ts_code", "symbol", "name", "exchange", "industry", "is_active", "updated_at"]]


def _hist(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    df = _retry(
        ak.stock_zh_a_hist,
        symbol=str(symbol).zfill(6),
        period="daily",
        start_date=start,
        end_date=end,
        adjust=adjust,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=_HIST_MAP)
    if "trade_date" not in df.columns:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def to_sina(ts_code: str) -> str:
    """600519.SH -> sh600519"""
    s = str(ts_code)
    code = s.split(".")[0].zfill(6)
    suffix = s.split(".")[1].upper() if "." in s else ""
    if not suffix:
        if code.startswith(("60", "68", "90")):
            suffix = "SH"
        elif code.startswith(("4", "8")):
            suffix = "BJ"
        else:
            suffix = "SZ"
    prefix = {"SH": "sh", "BJ": "bj"}.get(suffix, "sz")
    return f"{prefix}{code}"


def fetch_adj_factor(symbol: str, trade_dates) -> pd.DataFrame:
    """复权因子（新浪 hfq-factor），返回与 trade_dates 对齐的逐日因子。

    为什么不用东财 hfq 价反推（曾经的做法，已废弃）：
    东财后复权序列不是"原始价 × 阶梯因子"，而是做过收益压缩处理——
    实测逐日收益与原始收益的差 gap 与当日涨跌幅相关性达 -0.91
    （涨得越多 gap 越负）。后果是：
      · 用 hfq_close/close 当因子，因子每天漂移 ±2%，无法区分分红与噪声
      · 累计收益被系统性低估（五粮液一年真实 -39%，东财复权序列 -28%）
      · 均线、突破、ATR 全部建立在失真价格上
    新浪接口返回的是事件表：只在除权日给一行，因子阶梯式跳变，
    实测除权日与独立推算完全吻合（茅台 2024-06-19、五粮液 2026-07-16）。

    实现要点：事件表用"向前取最近一个事件"的方式展开到每个交易日，
    这样除权日当天即使用新因子（除权后价格 × 新因子 = 除权前价格 × 旧因子）。
    """
    td = pd.to_datetime(pd.Series(list(trade_dates)))
    if td.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
    try:
        ev = _retry(
            ak.stock_zh_a_daily,
            symbol=to_sina(symbol),
            start_date="19900101",
            end_date="20991231",
            adjust="hfq-factor",
        )
    except Exception:
        ev = None
    if ev is None or ev.empty or "hfq_factor" not in ev.columns:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])

    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["hfq_factor"] = pd.to_numeric(ev["hfq_factor"], errors="coerce")
    ev = ev.dropna(subset=["hfq_factor"]).sort_values("date")
    ev = ev[ev["hfq_factor"] > 0]
    if ev.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
    s = pd.Series(ev["hfq_factor"].values, index=ev["date"])
    s = s[~s.index.duplicated(keep="last")]

    idx = pd.DatetimeIndex(td.dt.normalize().unique())
    merged = s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)
    merged = merged.fillna(1.0)
    return pd.DataFrame({
        "ts_code": to_ts_code(symbol),
        "trade_date": pd.to_datetime(merged.index).date,
        "adj_factor": merged.values,
    }).sort_values("trade_date").reset_index(drop=True)


def fetch_daily(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """不复权日线 + 复权因子。

    返回 (bar, factor)：
      bar    列见 BAR_COLS，价格均为不复权原始价
      factor 列 [ts_code, trade_date, adj_factor]，price * factor = 后复权价
    """
    raw = _hist(symbol, start, end, "")
    if raw.empty:
        return (pd.DataFrame(columns=BAR_COLS),
                pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"]))

    raw["ts_code"] = to_ts_code(symbol)
    for c in ("open", "close", "high", "low", "vol", "amount", "amplitude",
              "pct_chg", "price_chg", "turnover"):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["pre_close"] = raw["close"] - raw["price_chg"]

    bar = raw[BAR_COLS].sort_values("trade_date").reset_index(drop=True)
    factor = fetch_adj_factor(symbol, bar["trade_date"])
    if factor.empty:
        # 新浪缺失时退化为东财 hfq 近似，宁可精度差也不让整只票没有因子
        hfq = _hist(symbol, start, end, "hfq")
        if not hfq.empty and "close" in hfq.columns:
            h = hfq[["trade_date", "close"]].rename(columns={"close": "hfq_close"})
            h["hfq_close"] = pd.to_numeric(h["hfq_close"], errors="coerce")
            m = bar[["ts_code", "trade_date", "close"]].merge(h, on="trade_date", how="left")
            ratio = (m["hfq_close"] / m["close"]).where(lambda x: x > 0)
            m["adj_factor"] = ratio.ffill().bfill().fillna(1.0)
            factor = m[["ts_code", "trade_date", "adj_factor"]].copy()
    return bar, factor.sort_values("trade_date").reset_index(drop=True)


def fetch_index(index_code: str, start: str, end: str) -> pd.DataFrame:
    """指数日线。index_code 传 6 位，如 000001 / 399006 / 000300。"""
    df = _retry(
        ak.index_zh_a_hist,
        symbol=str(index_code).zfill(6),
        period="daily",
        start_date=start,
        end_date=end,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=_INDEX_MAP)
    df["index_code"] = str(index_code).zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for c in ("open", "close", "high", "low", "vol", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["index_code", "trade_date", "open", "high", "low", "close",
               "vol", "amount"]].sort_values("trade_date")


def _disclosure_deadline(end_date: dt.date) -> dt.date:
    """法定披露截止日（保守推算，用于 point-in-time 可见性判断）。

    一季报 4-30 / 半年报 8-31 / 三季报 10-31 / 年报次年 4-30
    """
    m, y = end_date.month, end_date.year
    if m == 3:
        return dt.date(y, 4, 30)
    if m == 6:
        return dt.date(y, 8, 31)
    if m == 9:
        return dt.date(y, 10, 31)
    return dt.date(y + 1, 4, 30)


def fetch_yjbb(period: str) -> pd.DataFrame:
    """东财业绩报表。

    注意「最新公告日期」字段不可直接用于 point-in-time：
    实测茅台 2025 三季报（end_date=2025-09-30）的该字段返回 2026-10-31，
    是数据源统一刷新后的"最新"日期，不是首次披露日。
    因此额外计算 visible_date：
        visible_date = min(ann_date, 法定截止日)，且 ann_date 必须晚于 end_date
    这样既不会提前用到未来数据，也能在公告日可信时取到更精确的值。
    """
    df = _retry(ak.stock_yjbb_em, date=period)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            "股票代码": "symbol",
            "每股收益": "eps",
            "营业总收入-营业总收入": "revenue",
            "营业总收入-同比增长": "revenue_yoy",
            "营业总收入-季度环比增长": "revenue_qoq",
            "净利润-净利润": "net_profit",
            "净利润-同比增长": "net_profit_yoy",
            "净利润-季度环比增长": "net_profit_qoq",
            "每股净资产": "bps",
            "净资产收益率": "roe",
            "每股经营现金流量": "ocf_ps",
            "销售毛利率": "gross_margin",
            "所处行业": "industry",
            "最新公告日期": "ann_date",
        }
    )
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["ts_code"] = df["symbol"].map(to_ts_code)
    df["end_date"] = pd.to_datetime(period, format="%Y%m%d").date()
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce").dt.date
    df["ingested_at"] = pd.Timestamp.now().date()

    deadline = _disclosure_deadline(df["end_date"].iloc[0])
    # 公告日可信（晚于报告期且不晚于法定截止日）才采用，否则退回截止日
    visible = [
        min(a, deadline) if (a is not None and a > df["end_date"].iloc[0] and a <= deadline)
        else deadline
        for a in df["ann_date"]
    ]
    df["visible_date"] = visible

    num_cols = ["eps", "revenue", "revenue_yoy", "revenue_qoq", "net_profit",
                "net_profit_yoy", "net_profit_qoq", "bps", "roe", "ocf_ps",
                "gross_margin"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cols = ["ts_code", "end_date", "ann_date", "visible_date", "ingested_at",
            "eps", "revenue", "revenue_yoy", "revenue_qoq", "net_profit",
            "net_profit_yoy", "net_profit_qoq", "bps", "roe", "ocf_ps",
            "gross_margin", "industry"]
    return df[cols].dropna(subset=["ts_code"])


def fetch_market_activity() -> pd.DataFrame:
    """当日市场活跃度：上涨/下跌/涨停/跌停家数等。"""
    df = _retry(ak.stock_market_activity_legu)
    df.columns = ["metric", "value"]
    df["metric"] = df["metric"].astype(str)
    return df
