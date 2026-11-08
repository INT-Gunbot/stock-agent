"""W2 持仓体检入口。

用法：
    python -m scripts.run_checkup
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import checkup as C
from src.config import REPORT_DIR
from src.db import connect
from src.query import latest_trade_date

DISCLAIMER = (
    "> **免责声明**：以上内容基于公开数据和量化分析，仅供参考，不构成投资建议。"
    "市场有风险，投资需谨慎。任何投资决策应结合个人风险承受能力、资金状况和投资目标"
    "独立判断，必要时咨询持牌专业机构。过往表现不预示未来收益。"
)


def render_md(f: dict) -> str:
    p = f["profile"]
    hold = f["holdings"]
    lines = []
    a = lines.append

    a(f"# 持仓体检 · {f['trade_date'].replace('-', '')}")
    a("")
    a(f"**市场档位：{p['regime']}**，建议总仓位上限 {p['position_cap']}%，"
      f"单票权重建议不超过 {p['concentration_limit']}%")
    a("")

    # 1 组合画像
    a("## 一、组合画像")
    a("")
    a(f"- 持仓 {p['positions']} 只，市值 {p['mv']:,.0f}，总盈亏 {p['pnl']:+,.0f}（{p['pnl_pct']:+.2f}%）")
    a(f"- 当日 {p['day_ret']:+.2f}%，近5日加权 {p.get('ret5_w'):+.2f}%"
      + (f"，相对沪深300 {p['excess5']:+.2f}pct" if p.get("excess5") is not None else ""))
    a(f"- 盈利/浮亏：{p['winners']}/{p['losers']}；MA20 上方 {p['above_ma20']} 只，"
      f"MA60 上方 {p['above_ma60']} 只")
    a(f"- 前三大权重合计 {p['top3_weight']:.0f}%，最大单票 {p['max_weight_name']} {p['max_weight']:.0f}%")
    if p["overweight"]:
        a(f"- **集中度预警**：{('、'.join(p['overweight']))} 权重超过 {p['concentration_limit']}%")
    a("")

    # 2 分层分级
    a("## 二、分层分级（建议 vs 你的标注）")
    a("")
    dist = Counter(hold["suggest"])
    a("自动分级结果：" + "，".join(f"{k} {v} 只" for k, v in dist.most_common()))
    a("")
    a("| 名称 | 你的标注 | 建议仓别 | 复权盈亏 | 权重 | 位置 |")
    a("|---|---|---|---|---|---|")
    for _, r in hold.sort_values("weight", ascending=False).iterrows():
        if r.get("error"):
            a(f"| {r['name']} | — | 数据缺失 | — | — | {r['error']} |")
            continue
        pos = f"{r['pos20']:.2f}" if r.get("pos20") is not None else "—"
        pnl = f"{r['pnl_pct_adj']:+.1f}%" if r.get("pnl_pct_adj") is not None else "—"
        mark = " ⚠️" if r.get("tag_conflict") else ""
        a(f"| {r['name']} | {r['tag'] or '—'} | **{r['suggest']}**{mark} | {pnl} "
          f"| {r['weight']:.0f}% | {pos} |")
    a("")
    conflicts = hold[hold.get("tag_conflict", False)]
    if len(conflicts):
        a("**标注冲突**：" + "、".join(
            f"{r['name']}（标 {r['tag']}，量化特征更像 {r['suggest']}）"
            for _, r in conflicts.iterrows())
          + "。tag 影响日报里的文案口径，建议统一。")
        a("")

    # 3 调整优先级
    a("## 三、调整优先级")
    a("")
    if not f["actions"]:
        a("无可评估持仓。")
    else:
        sev_label = {3: "**今天**", 2: "**本周**", 0: "挂观察"}
        for x in f["actions"]:
            label = sev_label.get(x["severity"])
            head = f"{label}｜{x['name']}" if label else x["name"]
            a(f"- {head}（{x['suggest']}）→ {x['action']}：{x['why']}")
    a("")

    # 4 与选股池交叉
    a("## 四、与选股池的交叉")
    a("")
    if f["picks_from"]:
        a(f"基于 {f['picks_from']} 的选股榜单，头部：{'、'.join(f['picks_top'])}。")
        a("持仓与榜单的关系已并入上面第三节的条目；榜单本身见 select 报告，"
          "它回答的是\"池子里有什么\"，这里回答的是\"手里的和池子里的谁先动\"。")
    else:
        a("当日尚无选股记录（先运行 `python -m scripts.run_select`），"
          "本节自动留空，不影响前三节结论。")
    a("")
    a("**已知数据缺口**：行业集中度（缺行业分类）、与大盘相关性（缺资金流口径）"
      "本版未评估，接入后补上。")
    a("")
    a(DISCLAIMER)
    return "\n".join(lines)


def main(argv=None) -> int:
    con = connect(read_only=True)
    try:
        td = latest_trade_date(con)
        f = C.build(con, td)
    finally:
        con.close()

    md = render_md(f)
    stamp = str(f["trade_date"]).replace("-", "")
    out = REPORT_DIR / f"checkup_{stamp}.md"
    out.write_text(md, encoding="utf-8")

    # 同时落一份结构化数据供 demo / 后续接入使用
    import json
    from src.config import DATA_DIR
    hold = f["holdings"]
    payload = {
        "trade_date": f["trade_date"],
        "profile": {k: (None if isinstance(v, float) and v != v else v)
                    for k, v in f["profile"].items() if k != "overweight"},
        "overweight": f["profile"].get("overweight") or [],
        "actions": f["actions"],
        "picks_from": f["picks_from"],
        "picks_top": f["picks_top"],
        "holdings": [{
            "name": r["name"], "ts_code": r["ts_code"], "tag": r.get("tag", ""),
            "suggest": r.get("suggest"), "weight": r.get("weight"),
            "pnl_pct_adj": r.get("pnl_pct_adj"), "pos20": r.get("pos20"),
            "above_ma20": r.get("above_ma20"), "above_ma60": r.get("above_ma60"),
            "close": r.get("close"), "mv": r.get("mv"),
            "tag_conflict": bool(r.get("tag_conflict")),
        } for _, r in hold.iterrows()],
    }
    (DATA_DIR / f"checkup_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(f"[W2] {f['profile']['positions']} 只持仓，问题仓 "
          f"{sum(1 for x in f['actions'] if x['severity'] == 3)} 只"
          f" -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
