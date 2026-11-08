"""建 RAG 索引：方法论文档 + 历史报告 + 东财公告（标题 / 标题+正文）。

用法：
    python -m scripts.build_index                 # 文档 + 报告 + 最近 10 天公告标题
    python -m scripts.build_index --days 30       # 抓 30 天公告
    python -m scripts.build_index --bodies        # 公告连正文一起抓（P0 类型）
    python -m scripts.build_index --bodies --days 5
    python -m scripts.build_index --no-notice     # 只索引本地文档，不联网
    python -m scripts.build_index --stats         # 只看索引现状

为什么公告要单独抓：它是这个系统里唯一真正"非结构化"的数据。
行情、财务都是行列分明的数字，SQL 就够了；公告标题是自然语言，
"某某公司发布了什么"没法用 WHERE 查出来，这才是 RAG 该上场的地方。

--bodies 与默认路径的区别：
    默认只抓标题（618 条/天，4 秒），回答"谁发了什么公告"；
    --bodies 额外抓 P0 类型的正文并抽关键段落（回答"回购了多少股""募资多少"）。
    正文走 src/notices.py，存在 data/notice.db，与索引库 rag.db 分离。
    年报季单日 2.3 万条公告，--bodies 会跑很久，别在盘中执行。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.agent import rag  # noqa: E402
from src.config import REPORT_DIR  # noqa: E402


def index_docs(verbose: bool = True) -> dict:
    """根目录的 .md 是方法论：口径定义、数据边界、架构决策。"""
    total = 0
    files = sorted(BASE.glob("*.md"))
    keep: set[str] = set()
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        chunks = rag.chunk_markdown(text, source=p.name, kind="methodology")
        for i, c in enumerate(chunks):
            c["doc_key"] = f"doc::{p.name}#{i}"
            keep.add(c["doc_key"])
        n = rag.upsert_chunks(chunks)
        total += n
        if verbose:
            print(f"  {p.name:34s} {len(chunks):3d} 块")
    # 文档变短后章节数会减少，多出来的 #N 块不会自己消失（源文件还在，
    # 按文件名对齐发现不了），必须按"本文件实际产出的块键"对齐
    pr = rag.prune_keys("methodology", keep)
    if verbose and pr["删除"]:
        print(f"  清理失效文档块 {pr['删除']} 个")
    return {"文件": len(files), "块": total, "清理": pr["删除"]}


def index_reports(verbose: bool = True) -> dict:
    """历史日报/选股/体检：回答"上周三日报说了什么"这类回溯问题。"""
    total, n_files = 0, 0
    files = sorted(REPORT_DIR.glob("*.md"))
    keep: set[str] = set()
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        # 从文件名取日期：daily_20261102.md
        stem = p.stem
        date = ""
        for part in stem.split("_"):
            if len(part) == 8 and part.isdigit():
                date = part
        chunks = rag.chunk_markdown(text, source=p.name, kind="report")
        for i, c in enumerate(chunks):
            c["doc_key"] = f"report::{p.name}#{i}"
            c["trade_date"] = date
            keep.add(c["doc_key"])
        total += rag.upsert_chunks(chunks)
        n_files += 1
        if verbose:
            print(f"  {p.name:34s} {len(chunks):3d} 块")
    pr = rag.prune_keys("report", keep)
    if verbose and pr["删除"]:
        print(f"  清理失效报告块 {pr['删除']} 个")
    return {"文件": n_files, "块": total, "清理": pr["删除"]}


def index_notices(days: int = 10, verbose: bool = True) -> dict:
    """东财公告标题。每天约 600 条，抓一天 3 秒左右。"""
    try:
        import akshare as ak
        import pandas as pd
    except Exception as e:
        return {"错误": f"akshare 不可用：{e}"}

    end = pd.Timestamp.now().normalize()
    dates = [(end - pd.Timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]
    total, got = 0, 0
    ok_dates: set[str] = set()
    keep: set[str] = set()
    for d in dates:
        try:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                df = ak.stock_notice_report(symbol="全部", date=d)
        except Exception as e:
            if verbose:
                print(f"  {d} 抓取失败 {type(e).__name__}")
            continue
        if df is None or df.empty:
            continue
        got += 1
        chunks = []
        for _, r in df.iterrows():
            code = str(r.get("代码") or "").strip()
            name = str(r.get("名称") or "").strip()
            title = str(r.get("公告标题") or "").strip()
            ntype = str(r.get("公告类型") or "").strip()
            if not title:
                continue
            # 公告只有标题，没有正文。把代码/名称/类型都拼进正文，
            # 这样"茅台有什么公告"能靠 名称+代码 命中，"减持公告"能靠类型命中。
            body = f"{name}({code}) {title} | 公告类型：{ntype} | 日期：{d}"
            key = f"notice::{d}::{code}::{title[:60]}"
            keep.add(key)
            chunks.append({
                "doc_key": key,
                "source": "东财公告", "kind": "notice", "title": title,
                "text": body, "ts_code": _to_ts(code), "trade_date": d,
                "url": str(r.get("网址") or ""),
            })
        total += rag.upsert_chunks(chunks)
        ok_dates.add(d)
        if verbose:
            print(f"  {d} {len(chunks):4d} 条")
    # 只清理"这次真的抓到了"的日期：抓失败的日期整段不碰
    pr = rag.prune_keys("notice", keep, dates=ok_dates)
    if verbose and pr["删除"]:
        print(f"  清理失效标题块 {pr['删除']} 个")
    return {"交易日": got, "块": total, "清理": pr["删除"]}


def index_notice_bodies(days: int = 3, prio_max: str = "P0", verbose: bool = True) -> dict:
    """公告标题 + 正文一次抓完，再抽关键段落进索引。

    不回看太多天：正文接口单条约 0.2-1 秒，P0 类型一天 200 多条，
    30 天就是 6000 条 × 0.3 秒 ≈ 30 分钟。历史回填应该单独跑、
    按日期一天一天来，失败了能续，而不是塞进每次建索引里。
    """
    from src import notices

    end = time.time()
    dates = [time.strftime("%Y%m%d", time.localtime(end - i * 86400)) for i in range(days)]
    agg = {"标题": 0, "抓正文": 0, "成功": 0, "字符": 0}
    for d in dates:
        try:
            r = notices.harvest(d, prio_max=prio_max, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  {d} 失败 {type(e).__name__}: {str(e)[:80]}")
            continue
        for k in agg:
            agg[k] += r.get(k, 0)
    if verbose:
        print(f"  合计 标题 {agg['标题']} / 正文 {agg['成功']}/{agg['抓正文']} 条，"
              f"{agg['字符'] / 1e4:.0f} 万字符")
    idx = notices.build_index(verbose=verbose)
    return {**agg, **idx}


def _to_ts(code: str) -> str:
    """6 位代码 → ts_code。北交所/科创板/主板按前缀判断市场。"""
    c = str(code).strip()
    if not c.isdigit() or len(c) != 6:
        return c
    if c.startswith(("60", "68", "90")):
        return f"{c}.SH"
    if c.startswith(("4", "8", "92")):
        return f"{c}.BJ"
    return f"{c}.SZ"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10, help="抓最近 N 天公告，默认 10")
    ap.add_argument("--no-notice", action="store_true", help="不抓公告")
    ap.add_argument("--bodies", action="store_true",
                    help="公告连正文一起抓（P0 类型），默认只看标题")
    ap.add_argument("--prio", default="P0", choices=["P0", "P1", "P2"],
                    help="正文抓取范围，默认 P0（减持/回购/定增/诉讼/业绩预告等）")
    ap.add_argument("--stats", action="store_true", help="只看索引现状")
    args = ap.parse_args(argv)

    if args.stats:
        st = rag.stats()
        print(f"块数 {st['块数']}  词项 {st['词项数']}  分类 {st['分类']}")
        print(f"建于 {st['建索引时间']}")
        from src import notices
        n = notices.stats()
        print(f"公告库 标题 {n['标题条数']} / 有正文 {n['有正文']} 条 "
              f"（正文 {n['正文字符'] / 1e4:.0f} 万字符 → 关键段 {n['关键段字符'] / 1e4:.1f} 万，"
              f"压缩到 {n['压缩比'] * 100:.1f}%）{n['库大小MB']} MB / "
              f"纯PDF无文本 {n['无文本PDF']} 条")
        p0 = n["P0覆盖"]
        print(f"P0覆盖 {p0['已抓']}/{p0['标题']}（{p0['完成率'] * 100:.1f}%），待抓 {p0['待抓']}")
        for d, v in p0["按日"].items():
            print(f"  {d}  P0 {v['P0']:3d}  待抓 {v['待抓']:3d}")
        return 0

    t0 = time.time()
    print("[1/3] 方法论文档")
    docs = index_docs()
    print("[2/3] 历史报告")
    reps = index_reports()
    nres = {"块": 0, "跳过": True}
    if not args.no_notice:
        if args.bodies:
            print(f"[3/3] 公告标题+正文（最近 {args.days} 天，{args.prio}）")
            nres = index_notice_bodies(days=args.days, prio_max=args.prio)
        else:
            print(f"[3/3] 公告标题（最近 {args.days} 天）")
            nres = index_notices(days=args.days)

    rag.mark_built()
    st = rag.stats()
    print(f"\n完成，{round(time.time() - t0, 1)}s")
    if args.bodies:
        print(f"  文档 {docs['文件']} 个 / 报告 {reps['文件']} 个 / "
              f"公告标题 {nres.get('标题', 0)} 条 / 正文 {nres.get('成功', 0)} 条")
    else:
        print(f"  文档 {docs['文件']} 个 / 报告 {reps['文件']} 个 / "
              f"公告 {nres.get('交易日', 0)} 天，新增 {nres.get('块', 0)} 块，"
              f"清理 {nres.get('清理', 0)} 块")
    print(f"  索引合计 {st['块数']} 块，{st['词项数']} 个词项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
