"""W2 持仓评价引擎。

与日报的分工：
- 日报回答"今天发生了什么变化"（高频、只报异动）
- 体检回答"这个组合现在健康吗、该先处理谁"（低频、给结论）

三个节点：
1. 组合画像：收益结构 + 与沪深300对比 + 集中度
2. 分层分级：按量化特征给每票建议仓别，与用户 tag 冲突时明确指出
3. 调整优先级：给出"先处理谁"，并与 W1 榜单交叉回答"卖谁买谁"
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import portfolio as P
from . import query as Q
from .config import DATA_DIR
from .market import build as market_build


def _index_ret(con, code: str, n: int = 5) -> float | None:
    df = Q.get_index(code, con=con)
    if len(df) < n + 1:
        return None
    df = df[df["trade_date"] <= df["trade_date"].max()]
    return float(df["close"].iloc[-1] / df["close"].iloc[-1 - n] - 1)


def _load_latest_picks(trade_date: str) -> list[dict] | None:
    """读最近一次选股记录（W2 只读 W1 的落盘结果，不调用其函数）。"""
    path = DATA_DIR / "recommendations.jsonl"
    if not path.exists():
        return None
    last = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    if not last:
        return None
    rec = json.loads(last)
    return rec if rec.get("trade_date") == trade_date else None


# ---------------------------------------------------------------- 节点 1
def profile(con, trade_date, mk: dict, hold: pd.DataFrame) -> dict:
    ok = hold[hold.get("error").isna()] if "error" in hold.columns else hold
    s = P.summary(hold)
    if not ok.empty:
        w = ok["weight"] / 100
        s["ret5_w"] = float((w * ok["ret5"]).sum(skipna=True))
    hs300 = _index_ret(con, "000300", 5)
    s["excess5"] = (s["ret5_w"] - hs300 * 100) if (s.get("ret5_w") is not None and hs300) else None
    s["regime"] = mk["regime"]["level"]
    s["position_cap"] = mk["regime"]["position_cap"]
    # 集中度预警：档位越保守，单票权重容忍度越低
    cap = s["position_cap"] or 60
    s["concentration_limit"] = round(cap * 0.5)
    s["overweight"] = ok[ok["weight"] > s["concentration_limit"]]["name"].tolist() if not ok.empty else []
    return s


# ---------------------------------------------------------------- 节点 2
def classify(hold: pd.DataFrame, prof: dict) -> pd.DataFrame:
    """给每只票建议仓别。规则透明可解释，不用模型。"""
    ok = hold

    def _cls(r) -> str:
        if r.get("error"):
            return "数据缺失"
        pnl, a20, a60 = r["pnl_pct_adj"], r["above_ma20"], r["above_ma60"]
        if (pnl is not None and pnl < -15 and a60 is False) or (pnl is not None and pnl < -25):
            return "问题仓"
        if a20 and a60 and pnl is not None and pnl > 0:
            return "核心仓"
        if (r.get("atr_pct") is not None and pnl is not None and pnl > -5
                and r.get("weight") is not None and r["weight"] >= 15):
            return "稳定器仓"
        return "观察仓"

    ok = ok.copy()
    ok["suggest"] = ok.apply(_cls, axis=1)
    ok["tag_conflict"] = ok.apply(
        lambda r: bool(r.get("tag")) and r["suggest"] not in ("数据缺失",)
        and r["tag"] != r["suggest"], axis=1)
    return ok


# ---------------------------------------------------------------- 节点 3
def actions(hold: pd.DataFrame, prof: dict, picks: list[dict] | None) -> list[dict]:
    """调整优先级。severity 3 = 今天就该处理，1 = 挂观察。"""
    pick_codes = {p["ts_code"] for p in picks} if picks else set()
    top_names = [p["name"] for p in picks[:3]] if picks else []
    out = []

    for _, r in hold.iterrows():
        if r.get("error"):
            continue
        act, sev, why = "持有", 0, []
        pnl = r["pnl_pct_adj"]

        if r["suggest"] == "问题仓":
            act, sev = "制定退出计划", 3
            why.append(f"复权浮亏 {pnl:.1f}% 且已跌破 MA60" if r["above_ma60"] is False
                       else f"复权浮亏 {pnl:.1f}%")
        elif prof.get("overweight") and r["name"] in prof["overweight"]:
            act, sev = "控制单票权重", 2
            why.append(f"权重 {r['weight']:.0f}%，超过震荡档建议的 {prof['concentration_limit']}%")
        elif r["suggest"] == "核心仓":
            why.append("趋势完好，跌破 MA20 前不动作")
        elif r["suggest"] == "稳定器仓":
            why.append("波动低于组合中位，作为底仓")

        if r["above_ma20"] is False and r["suggest"] != "问题仓":
            why.append("短线在 MA20 下方，暂不加仓")

        in_pool = r["ts_code"] in pick_codes
        if in_pool:
            why.append("仍在当前选股强势池")
        elif r["suggest"] == "问题仓":
            alt = f"，W1 头部候选：{'、'.join(top_names)}" if top_names else ""
            why.append(f"不在当前强势池{alt}")

        out.append({
            "name": r["name"], "ts_code": r["ts_code"], "tag": r.get("tag", ""),
            "suggest": r["suggest"], "action": act, "severity": sev,
            "weight": r["weight"], "pnl_pct_adj": pnl, "why": "；".join(why),
            "tag_conflict": bool(r.get("tag_conflict")),
        })

    out.sort(key=lambda x: (-x["severity"], -(x["weight"] or 0)))
    return out


def build(con, trade_date) -> dict:
    mk = market_build(con, trade_date)
    hold = P.enrich(con, trade_date)
    prof = profile(con, trade_date, mk, hold)
    hold = classify(hold, prof)
    picks = _load_latest_picks(str(trade_date)[:10] if not isinstance(trade_date, str) else trade_date)
    acts = actions(hold, prof, picks["picks"] if picks else None)
    return {
        "trade_date": str(trade_date)[:10],
        "profile": prof,
        "holdings": hold,
        "actions": acts,
        "picks_from": picks["trade_date"] if picks else None,
        "picks_top": [p["name"] for p in picks["picks"][:5]] if picks else [],
    }
