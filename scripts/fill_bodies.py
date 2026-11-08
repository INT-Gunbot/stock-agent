"""补齐公告正文：重试失败项 + 抓当天新增 + 重建索引。

用法：
    python -m scripts.fill_bodies                    # 今天 + 昨天
    python -m scripts.fill_bodies 20261103           # 指定日期
    python -m scripts.fill_bodies 20261101 20261103  # 多天
    python -m scripts.fill_bodies --retry            # 只重试 ok=0 的失败项
    python -m scripts.fill_bodies --dry-run          # 只看还差多少，不抓

为什么要单独一个脚本：`harvest` 会跳过"库里已有 art_code"的公告，包括
ok=0（抓过但空/失败）那批——这是防死循环重试的设计，代价是失败项永远不自愈。
盘中公告又是持续新增的，10 点跑过一次 harvest，11 点还有两三百条新的 P0。
所以日常补数需要"清掉失败标记 → 重抓 → 增量入索引"这一整条。

失败项只在 `--retry` 或默认路径里清，清之前备份 art_code + 标题，
抓完仍失败的会重新写回 ok=0，不会无限重试同一批。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src import notices  # noqa: E402


def _con() -> sqlite3.Connection:
    return sqlite3.connect(str(notices.DB), timeout=10.0)


def failed_count() -> int:
    con = _con()
    try:
        return con.execute("SELECT count(*) FROM notice WHERE ok=0").fetchone()[0]
    finally:
        con.close()


def drop_failed(verbose: bool = True) -> int:
    """删掉 ok=0 的行，让 harvest 重新尝试。返回删除条数。"""
    con = _con()
    try:
        rows = con.execute("SELECT art_code, name, title FROM notice WHERE ok=0").fetchall()
        con.execute("DELETE FROM notice WHERE ok=0")
        con.commit()
    finally:
        con.close()
    if verbose:
        for a, n, t in rows:
            print(f"  重试 {n} {t[:40]}")
        if rows:
            print(f"  清掉 {len(rows)} 条失败标记")
    return len(rows)


def pending(date: str) -> dict:
    """某天还差多少条正文：P0 标题数 - 已抓（含成功与失败）。"""
    con = _con()
    try:
        rows = con.execute(
            "SELECT t.art_code, t.ntype, t.title FROM title t "
            "WHERE t.trade_date = ?", [date]).fetchall()
        have = {r[0] for r in con.execute("SELECT art_code FROM notice").fetchall()}
    finally:
        con.close()
    todo = [r for r in rows if notices.priority(r[1] or "", r[2] or "") == "P0"]
    left = [r for r in todo if r[0] not in have]
    return {"日期": date, "标题": len(rows), "P0": len(todo), "待抓": len(left)}


def run(dates: list[str], retry: bool = True, workers: int = 3,
        verbose: bool = True) -> dict:
    t0 = time.time()
    if retry:
        drop_failed(verbose=verbose)
    agg = {"日期数": len(dates), "P0": 0, "待抓": 0, "成功": 0, "空": 0,
           "失败": 0, "限流": 0, "字符": 0}
    for d in dates:
        p = pending(d)
        if verbose:
            print(f"[{d}] 标题 {p['标题']} / P0 {p['P0']} / 待抓 {p['待抓']}")
        if not p["待抓"]:
            # 没有待抓也要跑一次 harvest：盘中会冒出新的 title
            pass
        r = notices.harvest(d, prio_max="P0", workers=workers, verbose=verbose)
        for k in ("成功", "空", "失败", "限流", "字符"):
            agg[k] += r.get(k, 0)
        agg["P0"] += p["P0"]
        agg["待抓"] += p["待抓"]
    idx = notices.build_index(verbose=verbose)
    agg.update({"新增索引": idx["新增"], "清理索引": idx["清理"],
                "耗时": round(time.time() - t0, 1)})
    if verbose:
        print(f"\n合计：抓 {agg['成功']} 成功 / {agg['空']} 空 / {agg['失败']} 失败"
              f"{' / ' + str(agg['限流']) + ' 限流' if agg['限流'] else ''}，"
              f"{agg['字符'] / 1e4:.1f} 万字符，新增索引 {idx['新增']} 块，"
              f"清理 {idx['清理']} 块，{agg['耗时']}s")
        n = notices.stats()
        print(f"公告库：标题 {n['标题条数']} / 正文 {n['有正文']} 条 "
              f"（{n['正文字符'] / 1e4:.0f} 万字符 → 关键段 {n['关键段字符'] / 1e4:.1f} 万）"
              f"{n['库大小MB']} MB；剩余失败 {failed_count()} 条")
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="*", help="要补的日期，默认今天+昨天")
    ap.add_argument("--retry", action="store_true", help="只重试 ok=0 的失败项，不抓新公告")
    ap.add_argument("--dry-run", action="store_true", help="只看还差多少")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)

    dates = args.dates or [time.strftime("%Y%m%d", time.localtime(time.time() - i * 86400))
                           for i in range(2)]

    if args.dry_run:
        out = [pending(d) for d in dates]
        print(json.dumps({"待抓": out, "失败": failed_count()}, ensure_ascii=False, indent=2))
        return 0

    if args.retry:
        n = drop_failed()
        print(f"清掉 {n} 条失败标记，跑索引重建")
        notices.build_index(verbose=True)
        return 0

    r = run(dates, retry=True, workers=args.workers)
    return 0 if not r["失败"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
