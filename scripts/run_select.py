"""W1 选股推荐入口。

用法：
    python -m scripts.run_select            # 用库内最新交易日，默认启用换手缓冲带
    python -m scripts.run_select --top 30
    python -m scripts.run_select --no-buffer   # 关掉缓冲带，全量换仓（对比用）
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.config import REPORT_DIR
from src.db import connect
from src.market import build as market_build
from src.query import latest_trade_date
from src.scoring import KEEP_MULT, load_held, run_select

DISCLAIMER = (
    "> **免责声明**：以上内容基于公开数据和量化分析，仅供参考，不构成投资建议。"
    "市场有风险，投资需谨慎。过往表现不预示未来收益。"
)

DATA_GAPS = (
    "行业分类、资金流、龙虎榜、质押/解禁/减持数据尚未接入："
    "因子 A 为量价代理，质押解禁类一票否决规则暂不生效。"
)


def render_md(rec: dict) -> str:
    lines = []
    a = lines.append
    td = rec["trade_date"].replace("-", "")
    a(f"# 选股推荐 · {td}")
    a("")
    a(f"**市场档位：{rec['regime']}**（决定本次因子权重与建议仓位）")
    a("")
    a(f"- 覆盖 {rec['universe']} 只 → 一票否决后剩 **{rec['pool']} 只** → 取前 {len(rec['picks'])}")
    b = rec.get("buffer") or {}
    if b and b.get("buffer", 1) > 1:
        a(f"- 换手缓冲带 ×{b['buffer']}：上期 **{b.get('keep', 0)}** 只继续持有，"
          f"新进 **{b.get('new', 0)}** 只（掉出前 {b.get('keep_limit', 0)} 名才卖）")
    a(f"- {DATA_GAPS}")
    a("")
    v = rec["veto"]
    veto_items = [f"{k} {v[k]}" for k in v if k not in ("初始池", "可用池")]
    if veto_items:
        a("**一票否决明细**：" + "，".join(veto_items))
        a("")
    a("| # | 代码 | 名称 | 现价 | 综合分 | 主线资金 | 基本面 | 技术位 | 20日涨幅 | 量比 | 亮点 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, p in enumerate(rec["picks"], 1):
        a(
            f"| {i} | {p['ts_code']} | {p['name']} | {p['close']:.2f} "
            f"| {p['score']:.3f} | {p['score_A']:.2f} | {p['score_B']:.2f} "
            f"| {p['score_C']:.2f} | {p['ret20']*100:+.1f}% | {p['vol_ratio']:.2f} "
            f"| {p['reason']} |"
        )
    a("")
    a("**怎么用这份榜单**：它是观察候选池，不是买入指令。进攻档看前 10，"
      "震荡档结合自身持仓补强，防守档优先看 score_B 高的票。任何一只进场前，"
      "先核对自己的仓位上限和这只票的位置。")
    a("")
    a(DISCLAIMER)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--buffer", type=float, default=KEEP_MULT,
                    help="换手缓冲带倍数：持仓掉出前 top×buffer 名才卖（1 等于关闭）")
    ap.add_argument("--no-buffer", action="store_true", help="关闭缓冲带，全量换仓")
    args = ap.parse_args(argv)

    keep_mult = 1.0 if args.no_buffer else args.buffer
    con = connect(read_only=True)
    try:
        td = latest_trade_date(con)
        mk = market_build(con, td)
        held = load_held(args.top) if keep_mult > 1 else None
        rec = run_select(con, td, mk["regime"], top_n=args.top,
                         keep_mult=keep_mult, held=held)
    finally:
        con.close()

    md = render_md(rec)
    out = REPORT_DIR / f"select_{str(rec['trade_date']).replace('-', '')}.md"
    out.write_text(md, encoding="utf-8")
    b = rec.get("buffer") or {}
    buf_note = (f"，缓冲带 ×{b.get('buffer')}（留 {b.get('keep')} / 新进 {b.get('new')}"
                f" / 出池卖 {b.get('sold_pool')}）" if keep_mult > 1 else "，未启用缓冲带")
    print(f"[W1] 档位={rec['regime']} 池 {rec['universe']}->{rec['pool']}，"
          f"Top{len(rec['picks'])} -> {out.name}{buf_note}，记录已追加 recommendations.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
