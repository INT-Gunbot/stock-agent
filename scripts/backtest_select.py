"""选股引擎历史回测：把 src/scoring.py 的同一套逻辑搬回历史时点，量出真实效果。

为什么要单独写一个回测，而不是直接读 recommendations.jsonl：
  线上只跑了 8 次、最新交易日 2026-11-02，前向收益还不够 3 天，无法评估。

设计要点（与 src/scoring.py 保持一致的地方，以及刻意不同的地方）：

  一致：
    - 因子定义、权重表、一票否决规则逐条照搬
    - 财务数据用 visible_date 做时点对齐（不用 end_date，避免未来函数）
    - 因子全部走截面分位（rank pct），不做绝对值加权

  刻意不同：
    - 财务表 visible_date 最早 2025-04-22，之前 score_B 全空，
      所以回测区间从 2025-05 起，不硬凑 3 年
    - 成交价用「信号次日开盘」而不是当日收盘：信号是收盘后算的，
      按当日收盘成交属于不可实现的前视收益
    - 基准用全市场等权（同期同口径），而非沪深300，
      这样衡量的是「选股是否创造了超越平均的水平」，不受大小盘风格干扰

用法：
    python scripts/backtest_select.py                # 默认周频、持有 5/10/20 日
    python scripts/backtest_select.py --every 10     # 双周频
    python scripts/backtest_select.py --start 2025-06-01 --end 2026-06-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring import select_with_buffer   # noqa: E402  缓冲带与线上共用同一实现

DB = ROOT / "data" / "stock.duckdb"
OUT = ROOT / "data" / "backtest_select.json"

# ---- 与 src/scoring.py 完全一致的参数 ----
WEIGHTS = {
    "进攻": {"A": 0.45, "B": 0.25, "C": 0.30},
    "震荡": {"A": 0.40, "B": 0.35, "C": 0.25},
    "防守": {"A": 0.25, "B": 0.50, "C": 0.25},
}
MIN_AMT20 = 5e7
MIN_LIST_DAYS = 60
TOP_N = 20
HOLD = (5, 10, 20)


# ------------------------------------------------------------------ 数据装载
def load_matrices(con) -> dict:
    """一次性把日线搬进内存的宽表（日期 × 股票），后续所有计算全向量化。

    3.9M 行逐日跑 UNIVERSE_SQL 要 10s+/次，130 次就是半小时；
    宽表只需 4.1M 个 float 单元格，一次载入后每次截取一行。
    """
    df = con.execute("""
        SELECT b.ts_code, b.trade_date, b.open, b.close, b.vol, b.amount, b.pct_chg,
               b.close * a.adj_factor AS adjc,
               b.open  * a.adj_factor AS adjo
        FROM daily_bar b JOIN adj_factor a USING (ts_code, trade_date)
        ORDER BY b.trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    def wide(col):
        return df.pivot(index="trade_date", columns="ts_code", values=col).sort_index()

    px = {"adjc": wide("adjc"), "adjo": wide("adjo"), "close": wide("close"),
          "vol": wide("vol"), "amount": wide("amount"), "pct_chg": wide("pct_chg")}

    adjc = px["adjc"]
    # 与 SQL 里 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW 等价
    px["ma20"] = adjc.rolling(20, min_periods=1).mean()
    px["ma60"] = adjc.rolling(60, min_periods=1).mean()
    px["lo20"] = adjc.rolling(20, min_periods=1).min()
    px["hi20"] = adjc.rolling(20, min_periods=1).max()
    px["aamt20"] = px["amount"].rolling(20, min_periods=1).mean()
    px["avol20"] = px["vol"].rolling(20, min_periods=1).mean()
    px["ret20"] = adjc / adjc.shift(20) - 1
    px["ret60"] = adjc / adjc.shift(60) - 1
    px["vol_ratio"] = px["vol"] / px["avol20"]
    # n_days：该股票到当天为止的交易日数，作为上市天数代理
    px["n_days"] = adjc.notna().cumsum()
    return px


def load_fina(con) -> pd.DataFrame:
    return con.execute("""
        SELECT ts_code, end_date, visible_date, roe, net_profit_yoy, net_profit, gross_margin
        FROM fina_yjbb WHERE visible_date IS NOT NULL
    """).df()


def fina_asof(fina: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """取 asof 当天已公告的最新一期财报（按 visible_date 过滤，杜绝未来函数）。"""
    sub = fina[fina["visible_date"] <= pd.Timestamp(asof)]
    if sub.empty:
        return pd.DataFrame(columns=["ts_code", "roe", "net_profit_yoy",
                                     "net_profit", "gross_margin"])
    sub = sub.sort_values(["ts_code", "end_date", "visible_date"])
    return sub.groupby("ts_code", as_index=False).last()[
        ["ts_code", "roe", "net_profit_yoy", "net_profit", "gross_margin"]]


def regime_at(px, idx_close, dates, i) -> str:
    """与 src/market.py 同一判定：指数 MA60 位置 + 全市场上涨家数占比。"""
    if i < 60 or "000300" not in idx_close.columns:
        return "震荡"
    col = idx_close["000300"]
    c = col.iloc[i]
    ma60 = col.iloc[max(0, i - 59):i + 1].mean()
    if pd.isna(c) or pd.isna(ma60):
        return "震荡"
    above = c > ma60
    pc = px["pct_chg"].iloc[i]
    up_ratio = float((pc > 0).sum() / pc.notna().sum() * 100) if pc.notna().sum() else 0.0
    if above and up_ratio > 60:
        return "进攻"
    if (not above) and up_ratio < 35:
        return "防守"
    return "震荡"


# ------------------------------------------------------------------ 单期截面
def cross_section(px, d, i, names, fina) -> pd.DataFrame:
    """截取第 i 个交易日的截面，复刻 compute_universe + apply_veto + score。"""
    cols = ["adjc", "close", "vol", "n_days", "ma20", "ma60", "lo20", "hi20",
            "aamt20", "vol_ratio", "ret20", "ret60"]
    df = pd.DataFrame({c: px[c].iloc[i] for c in cols})
    df.index.name = "ts_code"
    df = df.reset_index()
    df = df[df["adjc"].notna()].copy()

    # pos20
    span = df["hi20"] - df["lo20"]
    df["pos20"] = np.where(span > 0, (df["adjc"] - df["lo20"]) / span, np.nan)
    df["name"] = df["ts_code"].map(names).fillna("")

    f = fina_asof(fina, d)
    df = df.merge(f, on="ts_code", how="left")
    return df.reset_index(drop=True)


def apply_veto(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats, n0 = {}, len(df)

    def cut(mask, reason):
        nonlocal df
        stats[reason] = int(mask.sum())
        df = df[~mask.fillna(False)]

    cut(df["name"].str.contains("ST|退", na=False), "ST 或退市风险")
    cut(df["ts_code"].str.endswith(".BJ"), "北交所（流动性）")
    cut(df["n_days"] < MIN_LIST_DAYS, f"上市不足 {MIN_LIST_DAYS} 日")
    cut(df["aamt20"] < MIN_AMT20, "20日均成交额过低")
    cut(df["vol"].fillna(0) <= 0, "当日停牌")
    cut(df["net_profit"].notna() & (df["net_profit"] < 0), "最新一期净利润为负")
    cut(df["net_profit_yoy"].notna() & (df["net_profit_yoy"] < -50), "净利同比下滑超 50%")
    cut(df["roe"].notna() & ((df["roe"] > 50) | (df["roe"] < -50)), "ROE 异常（口径可疑）")
    cut(df["net_profit_yoy"].notna() & (df["net_profit_yoy"] > 500), "净利同比基数失真")
    stats["初始池"], stats["可用池"] = n0, len(df)
    return df.reset_index(drop=True), stats


def score(df: pd.DataFrame, level: str) -> pd.DataFrame:
    w = WEIGHTS.get(level, WEIGHTS["震荡"])
    r = lambda s, asc=True: s.rank(pct=True, ascending=asc)

    df["score_A"] = 0.5 * r(df["ret20"]) + 0.3 * r(df["vol_ratio"]) + 0.2 * r(df["aamt20"])
    df["score_B"] = 0.4 * r(df["roe"]) + 0.4 * r(df["net_profit_yoy"]) + 0.2 * r(df["gross_margin"])
    c_pos = r(df["pos20"])
    c_ma60 = (df["adjc"] > df["ma60"]).astype(float)
    c_ma20 = (df["adjc"] > df["ma20"]).astype(float)
    df["score_C"] = 0.5 * c_pos + 0.25 * c_ma60 + 0.25 * c_ma20
    if level == "防守":
        df["score_C"] = 0.5 * r(-df["pos20"]) + 0.25 * c_ma60 + 0.25 * c_ma20

    df["score"] = (w["A"] * df["score_A"].fillna(df["score_A"].median())
                   + w["B"] * df["score_B"]
                   + w["C"] * df["score_C"])
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ 前向收益
def forward_returns(adjo: pd.DataFrame, dates, i, codes, holds=HOLD):
    """信号次日开盘买入，持有 h 个交易日后开盘卖出。

    返回 {h: Series(index=code)}。买入日不可得（停牌/退市）则为 NaN。
    """
    out = {}
    if i + 1 >= len(dates):
        return {h: pd.Series(dtype=float) for h in holds}
    buy_idx = i + 1
    buy = adjo.iloc[buy_idx]
    for h in holds:
        sell_idx = buy_idx + h
        if sell_idx >= len(dates):
            out[h] = pd.Series(index=codes, dtype=float)
            continue
        sell = adjo.iloc[sell_idx]
        out[h] = (sell / buy - 1).reindex(codes)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-05-06")
    ap.add_argument("--end", default="2026-09-25")
    ap.add_argument("--every", type=int, default=5, help="换仓间隔（交易日）")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--cost", type=float, default=0.003,
                    help="单次买卖往返成本（佣金+印花税+过户费+滑点），默认 0.3%%")
    ap.add_argument("--buffer", type=float, default=2.0,
                    help="换手缓冲带倍数：持仓掉出前 top×buffer 名才卖（1 等于关闭，全量换仓）")
    ap.add_argument("--out", default=None, help="结果文件路径，默认 data/backtest_select.json")
    args = ap.parse_args(argv)

    con = duckdb.connect(str(DB), read_only=True)
    print("载入日线矩阵 ...")
    px = load_matrices(con)
    fina = load_fina(con)
    names = dict(con.execute("SELECT ts_code, name FROM dim_stock").fetchall())
    names = {k: str(v or "").replace(" ", "") for k, v in names.items()}
    idx_close = con.execute(
        "SELECT trade_date, index_code, close FROM index_bar"
    ).df()
    idx_close["trade_date"] = pd.to_datetime(idx_close["trade_date"])
    idx_close = idx_close.pivot(index="trade_date", columns="index_code",
                                values="close").sort_index()

    dates = px["adjc"].index
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    cand = [i for i, d in enumerate(dates) if start <= d <= end]
    rebal = list(range(cand[0], cand[-1] + 1, args.every))
    print(f"换仓 {len(rebal)} 次｜{dates[rebal[0]].date()} → {dates[rebal[-1]].date()}")

    periods, prev_hold = [], []
    # 净值必须按「持有到下次换仓」的真实区间计，否则换仓间隔大于 5 日时
    # 会漏掉中间空仓的日子，年化被系统性算错
    holds = tuple(sorted(set(HOLD) | {args.every}))
    for n, i in enumerate(rebal, 1):
        d = dates[i]
        level = regime_at(px, idx_close, dates, i)
        cs = cross_section(px, d, i, names, fina)
        pool, veto = apply_veto(cs)
        ranked = score(pool, level)
        # 缓冲带：先看上一期持仓还剩几只合格，再用新票补空位。
        # 与 src/scoring.py 的线上实现共用同一个函数，避免两套逻辑漂移。
        picks, buf = select_with_buffer(ranked, prev_hold, args.top, args.buffer)

        fwd_pick = forward_returns(px["adjo"], dates, i, picks["ts_code"].tolist(), holds)
        mkt_codes = ranked["ts_code"].tolist()          # 可用池等权作基准，同口径
        fwd_mkt = forward_returns(px["adjo"], dates, i, mkt_codes, holds)

        # 全池 IC：因子值 vs 前向收益的 Spearman
        ic = {}
        for f in ("score", "score_A", "score_B", "score_C"):
            s = ranked.set_index("ts_code")[f]
            for h in HOLD:
                r = fwd_mkt[h]
                pair = pd.concat([s, r], axis=1, keys=["f", "r"]).dropna()
                ic[f"{f}_{h}"] = (pair["f"].rank().corr(pair["r"].rank())
                                  if len(pair) > 100 else np.nan)

        # 十分层：验证单调性。第 10 层 = 得分最高的 10%
        dec = {}
        s = ranked.set_index("ts_code")["score"].dropna()
        s_rank = s.rank(pct=True)                      # 升序分位：最高分 → 1.0
        grp = np.clip((s_rank * 10).astype("int64"), 0, 9) + 1   # 1..10，10 最高
        for h in (20,):
            r = fwd_mkt[h].reindex(s.index)
            dec[h] = {int(k): float(r[grp == k].mean())
                      for k in range(1, 11) if (grp == k).any()}

        rec = {
            "trade_date": str(d.date()), "regime": level,
            "universe": veto["初始池"], "pool": veto["可用池"],
            "buffer": buf,
            "picks": [{"ts_code": r["ts_code"], "name": r["name"], "score": round(r["score"], 4)}
                      for _, r in picks.iterrows()],
            "pick_ret": {}, "mkt_ret": {}, "excess": {},
            "hit": {}, "ic": {k: (None if pd.isna(v) else round(v, 4)) for k, v in ic.items()},
            "decile": dec.get(20, {}),
        }
        for h in HOLD:
            pr, mr = fwd_pick[h], fwd_mkt[h]
            rec["pick_ret"][h] = None if pr.dropna().empty else round(float(pr.mean()), 5)
            rec["mkt_ret"][h] = None if mr.dropna().empty else round(float(mr.mean()), 5)
            if rec["pick_ret"][h] is not None and rec["mkt_ret"][h] is not None:
                rec["excess"][h] = round(rec["pick_ret"][h] - rec["mkt_ret"][h], 5)
                # 个股胜率：逐只跑赢「等权基准均值」。基准必须用标量，
                # 若与逐股收益序列相减，同一只票恒为 0。
                rec["hit"][h] = round(float((pr.dropna() > rec["mkt_ret"][h]).mean()), 4)
            else:
                rec["excess"][h] = rec["hit"][h] = None

        # 随机组合对照：从同一可用池里随机抽 20 只，重复 300 次。
        # 若 Top20 的超额落在随机分布的中间，说明「选出来」这件事本身没有信息量。
        r20 = fwd_mkt[20].dropna()
        if len(r20) > args.top * 3 and rec["excess"].get(20) is not None:
            rng = np.random.default_rng(20261106 + i)
            arr = r20.to_numpy()
            # 每行随机排列，取前 top 个下标 → 300 组不放回随机组合
            idx = np.argsort(rng.random((300, arr.size)), axis=1)[:, :args.top]
            rnd_ex = arr[idx].mean(axis=1) - arr.mean()
            rec["rand"] = {
                "mean_pct": round(float(rnd_ex.mean()) * 100, 4),
                "std_pct": round(float(rnd_ex.std()) * 100, 4),
                "pick_pctile": round(float((rnd_ex < rec["excess"][20]).mean()), 3),
            }
        else:
            rec["rand"] = None

        if prev_hold:
            rec["turnover"] = round(1 - len(set(prev_hold) & {p["ts_code"] for p in rec["picks"]})
                                    / args.top, 4)
        else:
            rec["turnover"] = 1.0
        prev_hold = [p["ts_code"] for p in rec["picks"]]

        # 扣费：换手部分按往返成本计。这一项经常被忽略，
        # 但周频 70% 换手在 A 股是致命的——年化成本可达 10 个百分点。
        rec["cost_pct"] = round(rec["turnover"] * args.cost * 100, 4)
        rec["pick_ret_net"] = {h: (None if rec["pick_ret"][h] is None
                                  else round(rec["pick_ret"][h] - rec["turnover"] * args.cost, 5))
                               for h in HOLD}
        rec["excess_net"] = {h: (None if rec["excess"][h] is None
                                 else round(rec["excess"][h] - rec["turnover"] * args.cost, 5))
                             for h in HOLD}
        # 净值区间收益：持有 exactly every 个交易日
        pe, me = fwd_pick.get(args.every), fwd_mkt.get(args.every)
        rec["nav_ret"] = None if pe is None or pe.dropna().empty else round(float(pe.mean()), 5)
        rec["nav_bench"] = None if me is None or me.dropna().empty else round(float(me.mean()), 5)
        rec["nav_net"] = (None if rec["nav_ret"] is None
                          else round(rec["nav_ret"] - rec["turnover"] * args.cost, 5))
        periods.append(rec)
        if n % 10 == 0 or n == len(rebal):
            print(f"  {n}/{len(rebal)}  {d.date()} {level} 池{veto['可用池']} "
                  f"超额20日 {rec['excess'][20]}")

    summary = summarize(periods, args.top, args.every)
    out_path = Path(args.out) if args.out else OUT
    out_path.write_text(json.dumps({"config": vars(args), "summary": summary,
                                    "periods": periods}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print("\n" + "=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n明细已写入 {out_path}")
    return 0


def summarize(periods, top_n, every=5) -> dict:
    def col(key, h):
        return pd.Series([p[key].get(h) if isinstance(p[key], dict) else p[key]
                          for p in periods], dtype=float).dropna()

    s = {"periods": len(periods), "top_n": top_n}
    for h in HOLD:
        ex, pk, mk = col("excess", h), col("pick_ret", h), col("mkt_ret", h)
        hits = col("hit", h)
        ex_net = col("excess_net", h)
        s[f"h{h}"] = {
            "pick_mean_pct": round(pk.mean() * 100, 2),
            "mkt_mean_pct": round(mk.mean() * 100, 2),
            "excess_mean_pct": round(ex.mean() * 100, 2),
            "excess_median_pct": round(ex.median() * 100, 2),
            "excess_std_pct": round(ex.std() * 100, 2),
            "excess_net_pct": round(ex_net.mean() * 100, 2),
            "period_winrate": round(float((ex > 0).mean()), 4),
            "period_winrate_net": round(float((ex_net > 0).mean()), 4),
            "stock_winrate": round(float(hits.mean()), 4),
            # 信息比率：年化超额 / 年化跟踪误差（按 250 交易日折频）
            "ir": round(float(ex.mean() / ex.std() * np.sqrt(250 / 5)), 2) if ex.std() else None,
            "t_stat": round(float(ex.mean() / (ex.std() / np.sqrt(len(ex)))), 2) if len(ex) > 2 else None,
        }
    # IC
    ic_rows = {}
    for f in ("score", "score_A", "score_B", "score_C"):
        for h in HOLD:
            v = pd.Series([p["ic"].get(f"{f}_{h}") for p in periods], dtype=float).dropna()
            if len(v) > 2:
                ic_rows[f"{f}_{h}"] = {
                    "ic_mean": round(v.mean(), 4), "ic_std": round(v.std(), 4),
                    "icir": round(v.mean() / v.std(), 3) if v.std() else None,
                    "ic_pos_rate": round(float((v > 0).mean()), 3), "n": len(v)}
    s["ic"] = ic_rows

    # 十分层平均超额（验证单调性）
    dec = {}
    for k in range(1, 11):
        vals = [p["decile"].get(str(k), p["decile"].get(k)) for p in periods]
        vals = [v for v in vals if v is not None]
        dec[k] = round(float(np.mean(vals)) * 100, 2) if vals else None
    s["decile_ret20_pct"] = dec

    # 累计净值：区间收益 = 持有到下次换仓，无空仓缺口
    rets = pd.Series([p.get("nav_ret") for p in periods], dtype=float).dropna()
    nets = pd.Series([p.get("nav_net") for p in periods], dtype=float).dropna()
    mkts = pd.Series([p.get("nav_bench") for p in periods], dtype=float).dropna()
    if len(rets) > 2:
        nav = (1 + rets).cumprod()
        nnav = (1 + nets).cumprod()
        mnav = (1 + mkts).cumprod()
        yrs = len(rets) * every / 250
        s["nav"] = {
            "years": round(yrs, 2),
            "strategy_total_pct": round(float(nav.iloc[-1] - 1) * 100, 2),
            "strategy_net_total_pct": round(float(nnav.iloc[-1] - 1) * 100, 2),
            "bench_total_pct": round(float(mnav.iloc[-1] - 1) * 100, 2),
            "strategy_annual_pct": round(float(nav.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
            "strategy_net_annual_pct": round(float(nnav.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
            "bench_annual_pct": round(float(mnav.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
            "strategy_maxdd_pct": round(float((nav / nav.cummax() - 1).min()) * 100, 2),
            "strategy_net_maxdd_pct": round(float((nnav / nnav.cummax() - 1).min()) * 100, 2),
            "bench_maxdd_pct": round(float((mnav / mnav.cummax() - 1).min()) * 100, 2),
        }
        s["cost_annual_drag_pct"] = round(
            float((nav.iloc[-1] - nnav.iloc[-1]) * 100 / yrs), 2)
    # 档位分解
    df = pd.DataFrame([{"regime": p["regime"], "ex": p["excess"].get(20),
                        "ex_net": p["excess_net"].get(20)} for p in periods])
    s["by_regime"] = {g: {"n": int(len(sub)),
                          "excess20_pct": round(sub["ex"].mean() * 100, 2),
                          "excess20_net_pct": round(sub["ex_net"].mean() * 100, 2)}
                      for g, sub in df.dropna().groupby("regime")}
    # 随机对照：Top20 超额在各期随机分布中的平均百分位。50 附近 = 无信息量
    pcts = pd.Series([p["rand"]["pick_pctile"] for p in periods if p.get("rand")],
                     dtype=float).dropna()
    if len(pcts):
        s["random_control"] = {
            "avg_percentile": round(float(pcts.mean()), 3),
            "share_above_90pct": round(float((pcts > 0.9).mean()), 3),
            "n_periods": int(len(pcts)),
        }
    tv = pd.Series([p.get("turnover") for p in periods], dtype=float).dropna()
    s["turnover_mean"] = round(float(tv.mean()), 4) if len(tv) else None
    s["turnover_annual_x"] = round(float(tv.mean()) * 250 / every, 1) if len(tv) else None
    s["rebalance_days"] = every
    # 缓冲带实际生效情况：留任多少、新进多少、因出池被强制的卖出有多少
    bg = [p["buffer"] for p in periods if p.get("buffer")]
    if bg:
        s["buffer"] = {
            "keep_mult": bg[0].get("buffer"),
            "keep_mean": round(float(np.mean([b["keep"] for b in bg])), 2),
            "new_mean": round(float(np.mean([b["new"] for b in bg])), 2),
            "sold_pool_mean": round(float(np.mean([b["sold_pool"] for b in bg])), 2),
            "sold_rank_mean": round(float(np.mean([b["sold_rank"] for b in bg])), 2),
        }
    return s


if __name__ == "__main__":
    raise SystemExit(main())
