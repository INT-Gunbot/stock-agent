"""技术指标：纯函数，输入 DataFrame，输出 DataFrame。

约定（很重要）：
- 所有价格类指标一律用 `adj_close`（后复权），不用原始 close。
  除权日原始价会跳空（10 送 10 直接腰斩），用它算均线会在除权后长期失真。
- 所有展示给用户看的"今日涨跌幅"用 `ret` 列（attach_return 修正过），
  不要用 `pct_chg`——除权日 pct_chg 是错的。
- 这些函数不查库、不读文件，只做计算，可单测、可重跑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_ma(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20, 60)) -> pd.DataFrame:
    """均线。基于 adj_close。"""
    out = df.copy()
    px = out["adj_close"] if "adj_close" in out.columns else out["close"]
    for w in windows:
        out[f"ma{w}"] = px.rolling(w, min_periods=max(2, w // 2)).mean()
    return out


def add_vol_ratio(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """量能比 = 当日成交量 / 过去 n 日均量（不含当日）。

    >2 视为显著放量，<0.5 视为显著缩量。
    用 shift(1) 排除当日，否则当日巨量会把自己的基准抬高，导致比例被低估。
    """
    out = df.copy()
    base = out["vol"].shift(1).rolling(n, min_periods=max(3, n // 2)).mean()
    out["vol_ratio"] = out["vol"] / base
    return out


def add_atr(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """ATR（真实波幅均值），用复权价算，输出占价格百分比。

    用途：判断"跌 3%"到底是异动还是日常波动。
    高波动票跌 3% 可能只是 0.5 个 ATR，低波动票跌 3% 已是 3 个 ATR。
    """
    out = df.copy()
    high = out.get("adj_high", out["high"])
    low = out.get("adj_low", out["low"])
    prev_close = out["adj_close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr"] = tr.rolling(n, min_periods=max(3, n // 2)).mean()
    out["atr_pct"] = out["atr"] / out["adj_close"] * 100
    return out


def add_position(df: pd.DataFrame, windows: tuple[int, ...] = (20, 60)) -> pd.DataFrame:
    """价格在区间内的相对位置。

    pos20 = 0 表示处于 20 日最低，1 表示处于 20 日最高。
    比"距高点百分比"更好用：不同波动率的票可以直接比较。
    """
    out = df.copy()
    px = out["adj_close"]
    for w in windows:
        hi = px.rolling(w, min_periods=max(3, w // 2)).max()
        lo = px.rolling(w, min_periods=max(3, w // 2)).min()
        span = (hi - lo).replace(0, np.nan)
        out[f"pos{w}"] = (px - lo) / span
        out[f"hi{w}"] = hi
        out[f"lo{w}"] = lo
    return out


def add_returns(df: pd.DataFrame, windows: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """区间收益率，基于复权价。"""
    out = df.copy()
    px = out["adj_close"]
    for w in windows:
        out[f"ret{w}"] = px / px.shift(w) - 1
    return out


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """一次性附加日报需要的全部指标。"""
    out = add_ma(df)
    out = add_vol_ratio(out)
    out = add_atr(out)
    out = add_position(out)
    out = add_returns(out)
    return out


def describe(df: pd.DataFrame, name: str = "") -> dict:
    """把最后一行压成一句结论所需的最小事实集。

    只返回"可验证的数字"，不含任何措辞。措辞交给上层模板，
    这样数字错误可以在 facts 层被单测拦截。
    """
    if df.empty:
        return {}
    r = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else r

    def g(k, default=None):
        v = r.get(k, default)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    ma20 = g("ma20")
    ma60 = g("ma60")
    px = g("adj_close")
    return {
        "name": name or g("ts_code", ""),
        "ts_code": g("ts_code", ""),
        "close": g("close"),
        "trade_date": str(g("trade_date", ""))[:10],
        "ret": g("ret"),
        "vol_ratio": g("vol_ratio"),
        "atr_pct": g("atr_pct"),
        "ma5": g("ma5"),
        "ma10": g("ma10"),
        "ma20": ma20,
        "ma60": ma60,
        "above_ma20": bool(px and ma20 and px > ma20),
        "above_ma60": bool(px and ma60 and px > ma60),
        "ma20_slope5": (ma20 / df["ma20"].iloc[-6] - 1)
        if len(df) > 5 and not pd.isna(df["ma20"].iloc[-6])
        else None,
        "pos20": g("pos20"),
        "pos60": g("pos60"),
        "hi20": g("hi20"),
        "lo20": g("lo20"),
        "ret5": g("ret5"),
        "ret20": g("ret20"),
        "ret60": g("ret60"),
        "is_exdiv": bool(r.get("is_exdiv", False)),
        "prev_close": prev.get("close"),
    }
