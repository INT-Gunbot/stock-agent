"""持仓层：读取 positions.csv，计算盈亏、权重、当日贡献。

为什么成本收益要用复权口径：
持仓期间若发生分红送转，原始成本价与现价的比较会失真。
例：10 送 10 后股价腰斩，成本 30 元、现价 15 元，看原始价是亏 50%，
实际是持平。正确算法是把成本也乘上买入日的复权因子再比。

`adj_cost = cost * factor(买入日)`，与 `adj_close` 同一基准，可直接相除。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import indicators as I
from . import query as Q
from .config import BASE_DIR

POSITIONS_FILE = BASE_DIR / "positions.csv"

REQUIRED_COLS = ["ts_code", "shares", "cost"]


def load(path: Path | str | None = None) -> pd.DataFrame:
    """读持仓表。缺失字段给出明确报错，而不是让 NaN 静默传播到日报里。"""
    p = Path(path) if path else POSITIONS_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"持仓表不存在: {p}\n请在项目根目录创建 positions.csv，"
            f"字段：ts_code,name,shares,cost,open_date,tag,note"
        )
    # utf-8-sig 兼容带 BOM 的文件（Excel 另存的 CSV 常带 BOM）
    df = pd.read_csv(p, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"positions.csv 缺少必需列: {missing}")
    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    bad = df[df["shares"].isna() | df["cost"].isna()]
    if len(bad):
        raise ValueError(f"持仓表存在非法股数/成本: {bad['ts_code'].tolist()}")
    for c in ("name", "tag", "note"):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")
    if "open_date" not in df.columns:
        df["open_date"] = None
    return df


def enrich(con, trade_date, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """附加现价、复权收益、指标、当日贡献。

    一次连接内批量处理，避免每只票各开一次连接。
    """
    holds = load() if df is None else df.copy()
    # 资产类型一次查全：1664 个代码放内存里做成员判断，比每只票各查一次表快，
    # 也避免在循环里再开连接（DuckDB 每 open 一次约 50ms）。
    etf_codes = {r[0] for r in con.execute("SELECT ts_code FROM etf_dim").fetchall()}
    rows = []
    for _, h in holds.iterrows():
        code = h["ts_code"]
        is_etf = code in etf_codes
        px = Q.get_price(code, adjust="hfq", con=con,
                         asset="etf" if is_etf else "stock")
        if px.empty:
            rows.append({**h.to_dict(), "error": "无行情数据",
                         "asset": "etf" if is_etf else "stock"})
            continue
        px = px[px["trade_date"] <= pd.to_datetime(trade_date)]
        if px.empty:
            rows.append({**h.to_dict(), "error": "该日之前无行情",
                         "asset": "etf" if is_etf else "stock"})
            continue

        px = I.add_all(Q.attach_return(px))
        r = px.iloc[-1]
        prev = px.iloc[-2] if len(px) > 1 else r

        # 买入日因子：open_date 早于数据起点时退化为区间首日（收益会略有偏差，标注之）
        f_now = float(r["adj_factor"])
        f_buy = f_now
        buy_note = None
        if h.get("open_date"):
            try:
                bd = pd.to_datetime(h["open_date"])
                if bd >= px["trade_date"].iloc[0]:
                    sub = px[px["trade_date"] >= bd]
                    if len(sub):
                        f_buy = float(sub.iloc[0]["adj_factor"])
                else:
                    f_buy = float(px["adj_factor"].iloc[0])
                    buy_note = "买入日早于数据起点，收益按区间首日估算"
            except Exception:
                buy_note = "买入日无法解析，按当前因子估算"

        adj_cost = float(h["cost"]) * f_buy
        adj_close = float(r["adj_close"])
        shares = float(h["shares"])
        close = float(r["close"])
        mv = close * shares
        cost_amt = float(h["cost"]) * shares
        day_ret = float(r["ret"]) if pd.notna(r["ret"]) else None
        prev_mv = float(prev["close"]) * shares

        # 昨日指标：用来区分"今天刚跌破"和"一直在下方"。
        # 日报只该报前者——否则一只跌破 MA20 两个月的票会连续 40 天出现在异动列表里，
        # 列表就退化成状态罗列，用户三周后就不看了。
        def _prev_val(col):
            if len(px) < 2:
                return None
            v = px[col].iloc[-2]
            return None if pd.isna(v) else float(v)

        prev_ma20 = _prev_val("ma20")
        prev_pos20 = _prev_val("pos20")
        prev_adj_close = float(prev["adj_close"]) if pd.notna(prev.get("adj_close")) else None

        def q(v):
            """复权价 -> 当前价格口径（等价于以今天为基准的前复权价）。

            指标计算必须用连续的后复权价（否则除权日跳空会污染均线），
            但展示给用户时必须换回与现价同尺度的价格——
            否则会出现"茅台 MA20 = 11700，现价 1439"这种没法看的数字。
            adj_close = close × factor，所以 adj_close / factor_today 就是前复权价。
            """
            if v is None:
                return None
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            if pd.isna(fv) or f_now == 0:
                return None
            return fv / f_now

        rows.append({
            "ts_code": code,
            "asset": "etf" if is_etf else "stock",
            "name": h.get("name") or code,
            "tag": h.get("tag", ""),
            "note": h.get("note", ""),
            "shares": shares,
            "cost": float(h["cost"]),
            "open_date": str(h.get("open_date") or "")[:10],
            "close": close,
            "mv": mv,
            "cost_amt": cost_amt,
            "pnl": mv - cost_amt,
            "pnl_pct": (mv / cost_amt - 1) * 100 if cost_amt else None,
            # 复权口径收益：分红送转不会导致误判
            "pnl_pct_adj": (adj_close / adj_cost - 1) * 100 if adj_cost else None,
            "day_ret": day_ret * 100 if day_ret is not None else None,
            "day_pnl": (day_ret or 0) * prev_mv if prev_mv else None,
            "vol_ratio": _f(r.get("vol_ratio")),
            "atr_pct": _f(r.get("atr_pct")),
            "ma5": q(r.get("ma5")),
            "ma10": q(r.get("ma10")),
            "ma20": q(r.get("ma20")),
            "ma60": q(r.get("ma60")),
            "above_ma20": bool(pd.notna(r.get("ma20")) and adj_close > r["ma20"]),
            "above_ma60": bool(pd.notna(r.get("ma60")) and adj_close > r["ma60"]),
            "prev_above_ma20": (prev_adj_close > prev_ma20)
            if (prev_adj_close is not None and prev_ma20 is not None) else None,
            "pos20": _f(r.get("pos20")),
            "prev_pos20": prev_pos20,
            "pos60": _f(r.get("pos60")),
            "hi20": q(r.get("hi20")),
            "lo20": q(r.get("lo20")),
            "ret5": _f(r.get("ret5")) * 100 if _f(r.get("ret5")) is not None else None,
            "ret20": _f(r.get("ret20")) * 100 if _f(r.get("ret20")) is not None else None,
            "ret60": _f(r.get("ret60")) * 100 if _f(r.get("ret60")) is not None else None,
            "is_exdiv": bool(r.get("is_exdiv", False)),
            "buy_note": buy_note,
            "error": None,
        })

    out = pd.DataFrame(rows)
    ok = out[out["error"].isna()] if "error" in out.columns else out
    total_mv = ok["mv"].sum() if len(ok) else 0
    if total_mv:
        out["weight"] = out["mv"] / total_mv * 100
        out["day_contrib"] = out["day_pnl"] / total_mv * 100
    return out


def _f(v):
    """NaN -> None，避免 JSON/模板里出现 nan。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def summary(df: pd.DataFrame) -> dict:
    """组合层面汇总。"""
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    if ok.empty:
        return {}
    mv = float(ok["mv"].sum())
    cost = float(ok["cost_amt"].sum())
    day_pnl = float(ok["day_pnl"].fillna(0).sum())
    prev_mv = mv - day_pnl
    w = ok.sort_values("weight", ascending=False)["weight"].tolist()
    return {
        "positions": len(ok),
        "mv": mv,
        "cost": cost,
        "pnl": mv - cost,
        "pnl_pct": (mv / cost - 1) * 100 if cost else None,
        "day_pnl": day_pnl,
        "day_ret": (day_pnl / prev_mv * 100) if prev_mv else None,
        "top3_weight": sum(w[:3]),
        "max_weight": w[0] if w else None,
        "max_weight_name": ok.sort_values("weight", ascending=False).iloc[0]["name"]
        if len(ok) else None,
        "winners": int((ok["pnl_pct"] > 0).sum()),
        "losers": int((ok["pnl_pct"] <= 0).sum()),
        "above_ma20": int(ok["above_ma20"].sum()),
        "above_ma60": int(ok["above_ma60"].sum()),
    }
