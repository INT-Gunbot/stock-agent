"""每日数据更新入口。

用法:
  python -m scripts.update_all                      # 增量：全市场日线 + 指数 + 市场活跃度
  python -m scripts.update_all --full               # 首次全量（约 3 年，全市场较慢）
  python -m scripts.update_all --codes 600519,000001
  python -m scripts.update_all --fina               # 附带更新最近 8 个季度财务
"""
import argparse
import sys
import time

from src.update import (recent_periods, update_daily, update_fina,
                        update_index, update_market_activity, update_stock_list)


def main():
    ap = argparse.ArgumentParser(description="炒股 agent 数据更新")
    ap.add_argument("--full", action="store_true", help="首次全量回溯")
    ap.add_argument("--codes", type=str, default=None, help="限定股票，逗号分隔")
    ap.add_argument("--fina", action="store_true", help="同时更新财务数据")
    ap.add_argument("--periods", type=str, default=None, help="指定报告期，如 20250930,20250630")
    ap.add_argument("--workers", type=int, default=4, help="并发数")
    ap.add_argument("--skip-dim", action="store_true", help="跳过股票列表更新")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_dim:
        update_stock_list()

    codes = args.codes.split(",") if args.codes else None
    update_daily(codes=codes, full=args.full, workers=args.workers)
    update_index(full=args.full)
    update_market_activity()

    if args.fina or args.periods:
        periods = args.periods.split(",") if args.periods else recent_periods(8)
        print(f"[财务] 报告期: {periods}")
        update_fina(periods)

    print(f"[完成] 耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
