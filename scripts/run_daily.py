"""生成交易日报。

用法：
    python -m scripts.run_daily                 # 用最新交易日
    python -m scripts.run_daily --date 2026-10-30
    python -m scripts.run_daily --update        # 先增量更新行情再生成
    python -m scripts.run_daily --update --slow # 强制逐票拉取（补历史用）
    python -m scripts.run_daily --print         # 同时在终端打印 Markdown

注意 DuckDB 单文件写入时独占，所以必须"先更新、再生成"，
不要并发跑 update 和 report。
"""
from __future__ import annotations

import argparse
import sys

from src.db import connect
from src.report import daily as D
from src.report import render as R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="交易日 YYYY-MM-DD，默认取库内最新")
    ap.add_argument("--update", action="store_true", help="生成前先增量更新行情")
    ap.add_argument("--slow", action="store_true", help="更新时强制逐票拉取（补历史缺口用）")
    ap.add_argument("--print", dest="do_print", action="store_true", help="打印 Markdown")
    args = ap.parse_args(argv)

    if args.update:
        from src import update as U
        from src.config import DEFAULT_WORKERS

        if args.slow:
            r = U.update_daily(full=False, workers=DEFAULT_WORKERS, verbose=False)
            print(f"[更新·逐票] 成功 {r['ok']} 只，失败 {len(r['fail'])} 只，新增 {r['new_bars']} 行")
        else:
            r = U.update_daily_auto(verbose=True)
            if not r.get("ok"):
                print(f"[更新] 跳过：{r.get('error')}")
                print("[更新] 日报仍用库内现有数据生成，日期见下方")

    con = connect(read_only=True)
    try:
        facts = D.build(con, trade_date=args.date)
    finally:
        con.close()

    md, htm = R.save(facts)
    print(f"交易日 {facts['meta']['trade_date']}")
    print(f"  Markdown: {md}")
    print(f"  HTML    : {htm}")
    if args.do_print:
        print()
        print(R.render_markdown(facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
