"""W1 选股引擎：截面三因子打分。

设计原则（与工作流设计文档 v1 一致）：
- 先定市场档位，再定因子权重
- 所有因子走截面分位数（rank），不用绝对值加权
- 一票否决在打分之前执行，被否决的票不进池

数据缺口的诚实处理：
- 行业分类、资金流、龙虎榜暂缺 → 因子 A 用量价代理（动量/量能/流动性），
  输出中明确标注"行业相对强弱未纳入"
- 质押、解禁、减持数据暂缺 → 对应的一票否决规则不生效，输出中列出缺口
"""
from __future__ import annotations

import json
import datetime as dt

import duckdb
import numpy as np
import pandas as pd

from .config import DATA_DIR

# ---------------------------------------------------------------- 因子权重
WEIGHTS = {
    "进攻": {"A": 0.45, "B": 0.25, "C": 0.30},
    "震荡": {"A": 0.40, "B": 0.35, "C": 0.25},
    "防守": {"A": 0.25, "B": 0.50, "C": 0.25},
}

MIN_AMT20 = 5e7       # 20 日均成交额下限（流动性）
MIN_LIST_DAYS = 60    # 上市天数下限（剔除新股）
TOP_N = 20

# 换手缓冲带：已持仓只要没掉出前 TOP_N * KEEP_MULT 名就继续持有。
# 1.0 等于关掉（每期全量换仓），2.0 意味着前 40 名以内不动。
KEEP_MULT = 2.0

UNIVERSE_SQL = """
WITH px AS (
    SELECT b.ts_code, b.trade_date, b.close, b.vol, b.amount,
           b.close * a.adj_factor AS adjc
    FROM daily_bar b
    JOIN adj_factor a USING (ts_code, trade_date)
    WHERE b.trade_date <= ?
),
w AS (
    SELECT ts_code, trade_date, close, vol, amount, adjc,
           count(*) OVER (PARTITION BY ts_code) AS n_days,
           row_number() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) AS rn,
           avg(adjc) OVER (PARTITION BY ts_code ORDER BY trade_date
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
           avg(adjc) OVER (PARTITION BY ts_code ORDER BY trade_date
               ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60,
           min(adjc) OVER (PARTITION BY ts_code ORDER BY trade_date
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS lo20,
           max(adjc) OVER (PARTITION BY ts_code ORDER BY trade_date
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS hi20,
           avg(vol) OVER (PARTITION BY ts_code ORDER BY trade_date
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS avol20,
           avg(amount) OVER (PARTITION BY ts_code ORDER BY trade_date
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS aamt20,
           lag(adjc, 20) OVER (PARTITION BY ts_code ORDER BY trade_date) AS adjc_20ago,
           lag(adjc, 60) OVER (PARTITION BY ts_code ORDER BY trade_date) AS adjc_60ago
    FROM px
)
SELECT l.ts_code, d.name, l.trade_date, l.close, l.vol, l.n_days, l.adjc,
       l.ma20, l.ma60, l.lo20, l.hi20, l.avol20, l.aamt20,
       l.vol / nullif(l.avol20, 0) AS vol_ratio,
       CASE WHEN l.hi20 > l.lo20
            THEN (l.adjc - l.lo20) / (l.hi20 - l.lo20) END AS pos20,
       l.adjc / nullif(l.adjc_20ago, 0) - 1 AS ret20,
       l.adjc / nullif(l.adjc_60ago, 0) - 1 AS ret60
FROM w l LEFT JOIN dim_stock d ON d.ts_code = l.ts_code
WHERE l.rn = 1
"""

FINA_SQL = """
SELECT ts_code, roe, net_profit_yoy, gross_margin, net_profit FROM (
    SELECT ts_code, roe, net_profit_yoy, gross_margin, net_profit,
           row_number() OVER (PARTITION BY ts_code ORDER BY end_date DESC) AS rn
    FROM fina_yjbb WHERE visible_date <= ?
) WHERE rn = 1
"""


def _rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    """截面分位（0~1），NaN 不参与排名，结果保持 NaN。"""
    return s.rank(pct=True, ascending=ascending)


def compute_universe(con: duckdb.DuckDBPyConnection, trade_date) -> pd.DataFrame:
    """一次 SQL 算出全市场截面指标，再在 pandas 里做打分。"""
    df = con.execute(UNIVERSE_SQL, [_as_date(trade_date)]).df()
    fina = con.execute(FINA_SQL, [_as_date(trade_date)]).df()
    df = df.merge(fina, on="ts_code", how="left")
    # 上游名称常带空格（如"五 粮 液"），展示前统一去掉
    df["name"] = df["name"].fillna("").str.replace(" ", "", regex=False)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _as_date(v):
    if isinstance(v, dt.date):
        return v
    return pd.to_datetime(v).date()


def apply_veto(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """一票否决。返回 (可用池, 各规则剔除统计)。"""
    stats: dict[str, int] = {}
    n0 = len(df)

    def cut(mask: pd.Series, reason: str) -> None:
        nonlocal df
        k = int(mask.sum())
        stats[reason] = k
        df = df[~mask.fillna(False)]

    cut(df["name"].str.contains("ST|退", na=False), "ST 或退市风险")
    cut(df["ts_code"].str.endswith(".BJ"), "北交所（流动性）")
    cut(df["n_days"] < MIN_LIST_DAYS, f"上市不足 {MIN_LIST_DAYS} 日")
    cut(df["aamt20"] < MIN_AMT20, "20日均成交额过低")
    cut(df["vol"].fillna(0) <= 0, "当日停牌")
    cut(df["net_profit"].notna() & (df["net_profit"] < 0), "最新一期净利润为负")
    cut(df["net_profit_yoy"].notna() & (df["net_profit_yoy"] < -50), "净利同比下滑超 50%")
    # 财务清洗：ROE > 50% 基本是上游口径错误（单季/累计混淆）；
    # 同比 > 500% 说明上年同期接近零，数字没有参考意义，反而会污染截面 rank
    cut(df["roe"].notna() & ((df["roe"] > 50) | (df["roe"] < -50)), "ROE 异常（口径可疑）")
    cut(df["net_profit_yoy"].notna() & (df["net_profit_yoy"] > 500), "净利同比基数失真")

    stats["初始池"] = n0
    stats["可用池"] = len(df)
    return df.reset_index(drop=True), stats


def score(df: pd.DataFrame, regime_level: str) -> pd.DataFrame:
    """三因子截面打分。权重按市场档位切换。"""
    w = WEIGHTS.get(regime_level, WEIGHTS["震荡"])

    # A 主线资金（量价代理）
    a_mom = _rank(df["ret20"])
    a_vol = _rank(df["vol_ratio"])
    a_amt = _rank(df["aamt20"])
    df["score_A"] = 0.5 * a_mom + 0.3 * a_vol + 0.2 * a_amt

    # B 基本面（fina_yjbb，缺财务的票此因子为 NaN）
    b_roe = _rank(df["roe"])
    b_np = _rank(df["net_profit_yoy"])
    b_gm = _rank(df["gross_margin"])
    df["score_B"] = 0.4 * b_roe + 0.4 * b_np + 0.2 * b_gm

    # C 技术位置
    c_pos = _rank(df["pos20"])
    c_ma60 = (df["adjc"] > df["ma60"]).astype(float)
    c_ma20 = (df["adjc"] > df["ma20"]).astype(float)
    df["score_C"] = 0.5 * c_pos + 0.25 * c_ma60 + 0.25 * c_ma20

    # 防守档反向使用位置因子：找跌透的低吸标的而不是强势股
    if regime_level == "防守":
        df["score_C"] = 0.5 * _rank(-df["pos20"]) + 0.25 * c_ma60 + 0.25 * c_ma20

    df["score"] = (w["A"] * df["score_A"].fillna(df["score_A"].median())
                   + w["B"] * df["score_B"]
                   + w["C"] * df["score_C"])
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _reason(r: pd.Series) -> str:
    """由强项拼推荐理由，只描述可验证事实，不做涨跌预测。"""
    parts = []
    ra = r.get("score_A")
    if pd.notna(ra) and ra >= 0.8:
        parts.append(f"20日动量与量能均处市场前 {max(5, int((1-ra)*100))}%")
    rb = r.get("score_B")
    if pd.notna(r.get("roe")):
        parts.append(f"ROE {r['roe']:.1f}%、净利同比 {r['net_profit_yoy']:+.1f}%")
    elif rb is not None and pd.isna(rb):
        parts.append("暂无财务数据，基本面未计入")
    if pd.notna(r.get("pos20")) and r["pos20"] >= 0.9:
        parts.append("处20日区间高位（右侧强势）")
    elif pd.notna(r.get("pos20")) and r["pos20"] <= 0.2:
        parts.append("处20日区间低位（左侧低吸型）")
    if pd.notna(r.get("ma60")) and r["adjc"] > r["ma60"] and pd.notna(r.get("ma20")) and r["adjc"] <= r["ma20"]:
        parts.append("长线在 MA60 上方、短线回踩 MA20")
    return "；".join(parts) if parts else "综合分位靠前"


def load_held(top_n: int = TOP_N) -> list[str]:
    """读 recommendations.jsonl 最后一条的持仓代码，作为缓冲带的比较基准。"""
    path = DATA_DIR / "recommendations.jsonl"
    if not path.exists():
        return []
    last = ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    if not last:
        return []
    try:
        return [p["ts_code"] for p in json.loads(last).get("picks", [])][:top_n]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def select_with_buffer(ranked: pd.DataFrame, held: list[str] | None = None,
                       top_n: int = TOP_N, keep_mult: float = KEEP_MULT):
    """带缓冲带的持仓选择：压掉无效换手。

    全量换仓的问题不是"选错了"，而是"换得太勤"：评分每期都有些微抖动，
    排名 18 和 22 之间没有本质差别，但全量换仓会把它当成买卖信号，
    每次换掉 70% 以上的仓位，成本直接吃掉超额。

    规则：已持仓只要同时满足两条就继续持有
      1) 仍在当期可用池内（没被否决、没停牌、流动性达标）
      2) 得分排名未掉出 top_n * keep_mult
    腾出的名额再按得分从高到低补足，最终持仓数仍为 top_n。

    注意：被剔除出池的票必须卖出，不能用缓冲带掩盖——那是真风险，不是噪声。
    """
    if not held or keep_mult <= 1 or top_n <= 0:
        top = ranked.head(top_n).copy()
        return top, {"keep": 0, "new": int(len(top)), "sold_pool": 0,
                     "sold_rank": 0, "turnover": 1.0,
                     "buffer": keep_mult, "keep_limit": int(top_n)}

    rank = {c: i for i, c in enumerate(ranked["ts_code"])}
    keep_limit = int(round(top_n * keep_mult))
    held = list(dict.fromkeys(held))          # 去重且保留原顺序

    sold_pool = [c for c in held if c not in rank]           # 出池：必须卖
    in_pool = [c for c in held if c in rank]
    kept = sorted([c for c in in_pool if rank[c] < keep_limit], key=rank.get)
    sold_rank = [c for c in in_pool if rank[c] >= keep_limit]  # 排名掉太远：卖
    kept = kept[:top_n]

    keep_set = set(kept)
    fills = [c for c in ranked["ts_code"] if c not in keep_set][: top_n - len(kept)]
    chosen = keep_set | set(fills)

    out = ranked[ranked["ts_code"].isin(chosen)].copy()
    stats = {
        "keep": len(kept),
        "new": len(fills),
        "sold_pool": len(sold_pool),
        "sold_rank": len(sold_rank),
        "turnover": round(len(fills) / top_n, 4) if top_n else 0.0,
        "buffer": keep_mult,
        "keep_limit": keep_limit,
    }
    return out, stats


def run_select(con: duckdb.DuckDBPyConnection, trade_date, regime: dict,
               top_n: int = TOP_N, keep_mult: float = KEEP_MULT,
               held: list[str] | None = None) -> dict:
    """执行选股并落盘 recommendations.jsonl。

    keep_mult > 1 时启用换手缓冲带，held 为上一期持仓；held 为空则退化为全量选股。
    """
    uni = compute_universe(con, trade_date)
    pool, veto_stats = apply_veto(uni)
    ranked = score(pool, regime.get("level", "震荡"))
    top, buf_stats = select_with_buffer(ranked, held, top_n, keep_mult)
    top = top.copy()
    top["reason"] = top.apply(_reason, axis=1)

    record = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "trade_date": str(_as_date(trade_date)),
        "regime": regime.get("level"),
        "universe": int(veto_stats["初始池"]),
        "pool": int(veto_stats["可用池"]),
        "veto": veto_stats,
        "buffer": buf_stats,
        "picks": top[["ts_code", "name", "close", "score", "score_A",
                      "score_B", "score_C", "ret20", "vol_ratio", "roe",
                      "net_profit_yoy", "pos20", "reason"]].round(4).to_dict("records"),
    }
    path = DATA_DIR / "recommendations.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
