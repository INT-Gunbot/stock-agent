"""公告正文仓库：全量标题 + 按需正文 + 关键段落抽取。

## 这个模块回答的问题

"把所有公告正文都存进数据库，现实吗？"

先看实测（2026-11-02 与 2026-04-28 真实抓取）：

| 指标 | 日常 | 年报季高峰 |
|---|---|---|
| 东财公告条数/天 | 618 | 22,868（2026-04-28） |
| 单条正文均值 | 3,202 字符 | 年报/审计报告可达数十万 |
| 正文接口并发8均摊 | 31 ms/条 | 同 |

推全年：约 45 万条 / 18 亿字符。

**存储不贵**，实测 120 篇 39.4 万字符的膨胀率（相对 UTF-8 原文 1.13 MB）：

| 存法 | 体积 | 倍数 |
|---|---|---|
| 纯文本表 | 0.93 MB | 0.8x |
| FTS5 `trigram` | 3.23 MB | 2.9x |
| FTS5 bigram 预切 | 2.07 MB | 1.8x |
| 手写 BM25 倒排（本项目 rag.py 的结构） | 6.89 MB | 6.1x |

按 1.8x 算，全年全量正文 + bigram 索引 ≈ **14 GB/年**。NVMe 上不是问题。

**检索才是问题**：45 万条 × 4000 字符 / 420 字每块 = 428 万块/年。
BM25 在 428 万块上的行为不是"慢"，是**退化**——"公告"这类词 IDF→0，
年报的 1000 个块会把一条 200 字的减持公告彻底淹没。
换句话说：全量索引不会让你查得更准，只会让你查得更糊。

## 所以分三层

- **L0 标题层**（`rag.db` 现有 23k 条）：全量、每天一次、只索引标题+类型。
  回答"谁发了什么公告""最近有哪些减持"。这一层永远保持轻。
- **L1 正文层**（本模块 `notice.db`）：按类型分级决定要不要抓正文。
  P0 抓全文，P1 只抓前 2 页，P2 不抓。原文 zlib 压缩存 BLOB，
  另存一份"关键段落"明文——检索和解压脱钩，热路径不碰 BLOB。
- **L2 索引层**：只有白名单类型的**关键段落**进 RAG，不是全文进。
  4000 字符压到 700 字符，块数降一个量级，IDF 才有区分度。

## 关键段落怎么抽

公告的行文是高度模板化的：一句 100 字的"保证真实准确完整"声明、
一行"证券代码/公告编号"、最后一段"特此公告 + 董事会 + 日期"。
这些占正文 20-30%，信息量为零，但会在 BM25 里反复拉高词频。

抽取规则按"可核查性"排序，而不是按位置：
1. 含**数字**的句子优先——金额、股数、比例、价格、日期，这些是公告的实质；
2. 含**行为动词**的句子次之——拟/决定/终止/签署/回购/减持/增持/质押/冻结/诉讼/立案；
3. 首段兜底——公告首段通常是"一句话说清干了什么"。

反过来，声明句、编号行、盖章行直接丢。
800 字符的预算下，这个顺序保住的都是能被引用的事实，而不是套话。
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config import DATA_DIR

DB = DATA_DIR / "notice.db"

# 东财正文接口单页上限 5000 字符，page_size 字段给的是总页数
PAGE_CHARS = 5000
MAX_PAGES = 40          # 年报也就 20-30 页，40 页兜底防死循环
KEY_CHARS = 700         # 关键段落预算：45 万条 × 700 字符 ≈ 0.9 GB/年

# ---------------------------------------------------------------- 类型分级
# 判定按"这个类型里有没有可交易的事实"。不是按公告长短——长的常常最没用。
_P0 = ("减持", "增持", "回购", "质押", "冻结", "业绩预告", "业绩快报", "预增", "预减",
       "重大合同", "中标", "诉讼", "仲裁", "停牌", "复牌", "立案", "处罚", "违规",
       "控制权", "要约", "收购", "重组", "定增", "向特定对象发行", "股权激励",
       "分红", "权益分派", "澄清", "风险警示", "退市", "关联交易", "担保", "对外投资")
_P1 = ("高管", "股东大会决议", "董事会决议", "监事会决议", "募集资金", "其他",
       "人事", "聘任", "辞职", "变更")
_P2 = ("独立董事述职", "述职报告", "保荐", "核查意见", "审计报告", "法律意见",
       "管理办法", "制度", "章程", "专项说明", "独立意见", "年报全文", "年度报告",
       "季度报告", "半年度报告", "内部控制", "社会责任", "专项审计")


def priority(ntype: str, title: str = "") -> str:
    """P0 抓全文 / P1 抓前两页 / P2 不抓。标题也算，很多类型字段是「其他」。"""
    s = f"{ntype or ''} {title or ''}"
    for kw in _P0:
        if kw in s:
            return "P0"
    for kw in _P2:
        if kw in s:
            return "P2"
    for kw in _P1:
        if kw in s:
            return "P1"
    return "P1"


# ---------------------------------------------------------------- 库
_SCHEMA = """
CREATE TABLE IF NOT EXISTS notice (
    art_code   TEXT PRIMARY KEY,     -- 东财 AN 编号，正文接口的唯一键
    ts_code    TEXT,
    name       TEXT,
    title      TEXT,
    ntype      TEXT,
    trade_date TEXT,
    url        TEXT,
    prio       TEXT,                 -- P0/P1/P2
    n_pages    INTEGER,              -- 正文实际页数（抓过才有）
    raw_chars  INTEGER,              -- 原文长度
    key_text   TEXT,                 -- 关键段落明文：检索热路径直接读，不解压
    body       BLOB,                 -- zlib(原文)：冷数据，用户要全文时才解压
    fetched_at TEXT,
    ok         INTEGER DEFAULT 1     -- 0 = 抓过但空/失败，避免反复重试
);
CREATE INDEX IF NOT EXISTS idx_notice_code ON notice(ts_code, trade_date);
CREATE INDEX IF NOT EXISTS idx_notice_date ON notice(trade_date);
CREATE INDEX IF NOT EXISTS idx_notice_prio ON notice(prio, trade_date);

-- 标题层：全量存，不等正文。这样"最近有哪些减持公告"不依赖抓正文
CREATE TABLE IF NOT EXISTS title (
    art_code   TEXT PRIMARY KEY,
    ts_code    TEXT,
    name       TEXT,
    title      TEXT,
    ntype      TEXT,
    trade_date TEXT,
    url        TEXT
);
CREATE INDEX IF NOT EXISTS idx_title_code ON title(ts_code, trade_date);
CREATE INDEX IF NOT EXISTS idx_title_date ON title(trade_date);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def _con() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB), timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(_SCHEMA)
    return con


# ---------------------------------------------------------------- 抓取
_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
_API = "https://np-cnotice-stock.eastmoney.com/api/content/ann"

# 东财正文接口会限流。实测：8 并发连抓 600 条后返回 **567**（不是 429），
# 响应体是一个 HTML 风控页而不是 JSON。这个状态码不在任何标准里，
# 但它出现后，同一个 IP 继续请求会持续 567，要等几分钟才恢复。
# 所以限流不能靠"失败了重试"——重试只会加重封禁。只能从源头限速。
QPS = 4.0
_RATE_LOCK = __import__("threading").Lock()
_rate_next = 0.0
_rate_pause = 0.0


class RateLimited(Exception):
    """被东财风控拦截。调用方应该退避而不是立刻重试。"""


def _throttle() -> None:
    """全局令牌桶：所有线程共享同一个下一次可发请求的时间点。"""
    global _rate_next
    with _RATE_LOCK:
        now = time.time()
        t = max(now, _rate_next, _rate_pause)
        _rate_next = t + 1.0 / QPS
    wait = t - time.time()
    if wait > 0:
        time.sleep(wait)


def _backoff(seconds: float) -> None:
    """全局暂停：一个线程撞上风控，所有线程一起停，否则其它线程还在送请求。"""
    global _rate_pause
    if _RATE_LOCK is not None:
        with _RATE_LOCK:
            _rate_pause = max(_rate_pause, time.time() + seconds)


def _session():
    import requests
    s = requests.Session()
    s.headers.update(_UA)
    return s


def art_of(url: str) -> str:
    """https://data.eastmoney.com/notices/detail/600519/AN2024xxx.html → AN2024xxx"""
    return str(url or "").rstrip("/").rsplit("/", 1)[-1].replace(".html", "")


def fetch_body(art_code: str, max_pages: int = MAX_PAGES, sess=None) -> dict:
    """翻页拿正文。返回 {"正文": str, "标题": str, "页数": int}。

    东财的 page_size 是**总页数**而不是每页字符数，这个字段名很有欺骗性——
    第一版按"每页 5000 字 × page_size"估算长度，结果把 13 万字的年报估成了 2500 万字。
    实际用法只能是：一直翻到某页返回不足 4500 字为止。

    超长公告要翻几十页，一页一次请求，是限流的主要风险点。
    所以每页都过令牌桶，不因为是"同一条公告的下一页"就免限速。
    """
    s = sess or _session()
    parts: list[str] = []
    title = ""
    for pi in range(1, max_pages + 1):
        _throttle()
        r = s.get(_API, params={"art_code": art_code, "client_source": "web",
                                "page_index": pi}, timeout=20)
        if r.status_code != 200 or not r.text.lstrip().startswith("{"):
            # 567 / 风控 HTML：不是这条公告有问题，是 IP 被限了
            raise RateLimited(f"HTTP {r.status_code}")
        d = (r.json() or {}).get("data") or {}
        if not d:
            break
        if pi == 1:
            title = d.get("notice_title") or ""
        c = d.get("notice_content") or ""
        parts.append(c)
        if len(c) < PAGE_CHARS * 0.9:
            break
    text = "".join(parts).strip()
    return {"正文": text, "标题": title, "页数": len(parts)}


def fetch_titles(date: str, sess=None) -> list[dict]:
    """某天的全量公告标题。年报季单日 2.3 万条、耗时 90 秒，别在盘中调。"""
    import contextlib
    import io

    import akshare as ak

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        df = ak.stock_notice_report(symbol="全部", date=date)
    if df is None or getattr(df, "empty", True):
        return []
    out = []
    for r in df.to_dict("records"):
        url = str(r.get("网址") or "").strip()
        title = str(r.get("公告标题") or "").strip()
        if not url or not title:
            continue
        out.append({
            "art_code": art_of(url),
            "ts_code": _to_ts(str(r.get("代码") or "")),
            "name": str(r.get("名称") or "").strip(),
            "title": title,
            "ntype": str(r.get("公告类型") or "").strip(),
            "trade_date": date,
            "url": url,
        })
    return out


def _to_ts(code: str) -> str:
    c = str(code).strip()
    if not c.isdigit() or len(c) != 6:
        return c
    if c.startswith(("60", "68", "90")):
        return f"{c}.SH"
    if c.startswith(("4", "8", "92")):
        return f"{c}.BJ"
    return f"{c}.SZ"


# ---------------------------------------------------------------- 关键段落
# 声明句：每个公告都有，措辞略有不同所以精确匹配抓不到，只能按特征拦
_NOISE = re.compile(
    r"(保证.{0,20}(真实|准确|完整)|没有虚假记载|不存在虚假记载|承担个别和连带|"
    r"连带法律责任|敬请.{0,6}投资者.{0,10}注意|特此公告|备查文件|"
    r"^证券代码|^证券简称|^公告编号|^股票代码|^本公司董事会|^董事会$|"
    # 法律条文复读机：每家公司的回购/减持公告都会抄一遍交易所规则，
    # 这些句子不含本公司任何事实，却因为"含条款序号"拿到加分，反复挤占预算
    r"规定的其他(情形|要求)|^（\d）.*(证券交易所|证监会)|"
    r"^\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*$)"
)
# 版头行：证券代码/简称/编号常挤在一行，整体丢掉
_HEADER = re.compile(r"^(证券代码|证券简称|公告编号|股票代码|股票简称)")
# 有数字 = 有事实。金额/股数/比例/价格/日期都在这条规则里
_NUM = re.compile(r"\d")
# 行为动词：不带动词的句子通常是小标题或过渡句
_ACT = re.compile(r"(拟|决定|同意|终止|签署|签订|回购|减持|增持|质押|冻结|解除|"
                  r"诉讼|仲裁|立案|处罚|变更|收购|转让|发行|授予|行权|分红|派息|"
                  r"停牌|复牌|中标|违约|逾期|担保|募集|增资|减资|清算|破产|重整)")


# PDF 转文本的两种典型垃圾：
# 目录行（"三、股东分红回报规划......40"）和表格截断碎片（引号/括号没配对的半句）
_TOC = re.compile(r"\.{3,}\s*\d+\s*$|…{2,}\s*\d+\s*$")
# 表格行的标志：列之间靠 4 个以上空格对齐
_COL = re.compile(r"\S[ \t\u3000]{4,}\S")


_ORDINAL = re.compile(r"^(?:\d{1,2}\s*[、.．]|[一二三四五六七八九十]{1,3}\s*[、.]|（[一二三四五六七八九十\d]{1,3}）)\s*")
# 表格行粘连：PDF 里相邻单元格的数字首尾相接，如 "1,810.472,263.092,909.69"
_GLUED = re.compile(r"\d\.\d{2}\d")


def _is_table_shard(s: str) -> bool:
    """表格转文本后的半句。判据是标点不配对——正常中文句子不会这样断。"""
    if _TOC.search(s):
        return True
    for a, b in (("“", "”"), ("（", "）"), ("《", "》"), ("【", "】")):
        if s.count(a) != s.count(b):
            return True
    # 三个以上小数被粘连，说明是横向表格而不是句子
    if len(_GLUED.findall(s)) >= 2:
        return True
    return False


def _sents(text: str) -> list[str]:
    """把正文切成"句子"，同时整块丢掉表格。

    ## 为什么不直接把所有空白压掉

    第一版 `re.sub(r"[ \\t\\u3000]+", "", t)` 把所有空格删了，代价是
    表格行塌成一行数字："1,810.472,263.092,909.69"——看起来像一句话，
    实际是三个单元格的金额连在一起。它还能通过"含数字"的打分，
    于是一份问询函的关键段落里混进了好几行这种垃圾。

    正确做法是**先按行判断，再决定拼不拼**：一行里出现两处以上 4 空格对齐，
    就是表格行，整行丢掉；连续的非表格行才拼回段落。
    东财正文的段落分隔是空行，所以这个判断很干净。
    """
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        para = " ".join(buf).strip()
        buf.clear()
        if not para:
            return
        # PDF 行尾断开后拼接会留下空格："扣非 归母净利润"。中文之间的空格是噪声，
        # 数字和中文之间的空格要留（"2,229.63 万元"读起来才不粘连）。
        para = re.sub(r"(?<=[\u4e00-\u9fff，。；：、（）“”‘’])\s+(?=[\u4e00-\u9fff，。；：、（）“”‘’])",
                      "", para)
        for s in re.split(r"(?<=[。；！？])", para):
            s = s.strip()
            if len(s) >= 10:
                out.append(s)

    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            flush()
            continue
        if len(_COL.findall(ln)) >= 2 or _TOC.search(s) or _HEADER.search(s):
            flush()          # 表格行/版头截断段落，自身不入库
            continue
        buf.append(s)
    flush()
    return out


def _bucket_of(i: int, n: int, buckets: int) -> int:
    return min(buckets - 1, int(i * buckets / max(n, 1)))


def key_paragraphs(text: str, max_chars: int = KEY_CHARS) -> str:
    """按「可核查性」抽关键段落，不是按位置。

    ## 为什么不能只按分数取前 N 句

    第一版取全局最高分，结果在一份 19.5 万字符的问询函回复上彻底失效：
    PDF 里重复的表格表头让同一句话出现几十次，700 字符的预算被
    「二、关于反馈意见"2．截至…其他应收款账面价」这一句的 18 个副本吃光，
    全文其余部分一个字都没进来。19 万字的公告，摘要只覆盖了第 3 页。

    修法两条：
    1. **句子去重**——归一化后前 24 字相同即视为同一句（表格表头、页眉都这样重复）；
    2. **分桶采样**——把文档按位置切成 5 段，每段独立打分、各拿 1/5 预算。
       长文档的关键信息不在开头：问询函的风险提示在结尾，年报的财务数据在中段。

    ## 为什么打分而不是过滤

    单纯"含数字就留"会把"2026 年 11 月 2 日"这种纯日期句也留下，攒满预算全是时间戳，
    真正说回购了多少股的句子反而被挤掉。所以数字、动词、长度、条款序号各自加权。
    """
    sents = [s for s in _sents(text) if not _NOISE.search(s)]
    if not sents:
        return ""

    # 去重：保留首次出现的位置，后面同句直接丢。
    # key 要剥掉"5、""（3）"这类序号——同一句话在"目录"和"正文"里序号不同，
    # 不剥就是两份，正好让重复句挤占预算。
    uniq: list[tuple[int, str]] = []
    seen: set[str] = set()
    for i, s in enumerate(sents):
        key = re.sub(r"\W", "", _ORDINAL.sub("", s))[:24]
        if key in seen:
            continue
        seen.add(key)
        if _is_table_shard(s):
            continue
        uniq.append((i, s))
    if not uniq:
        return ""

    n = len(sents)
    buckets = 5 if n >= 20 else 1
    quota = max_chars / buckets
    picked: list[tuple[int, str]] = []

    for b in range(buckets):
        group = []
        for i, s in uniq:
            if _bucket_of(i, n, buckets) != b:
                continue
            score = 0.0
            nums = len(_NUM.findall(s))
            if nums:
                score += 2.0 + min(nums, 6) * 0.5   # 数字越多越像实质条款，封顶防长表格霸榜
            if _ACT.search(s):
                score += 1.5
            if 20 <= len(s) <= 200:
                score += 0.8                        # 太短没信息，太长是罗列
            if i < 3:
                score += 0.6                        # 首段通常是"一句话结论"
            if re.search(r"[一二三四五六七八九十]、|（[一二三四五]）", s):
                score += 0.3                        # 条款序号
            group.append((score, i, s))
        group.sort(key=lambda x: -x[0])
        size = 0
        for score, i, s in group:
            if size + len(s) > quota * 1.15:        # 允许 15% 溢出，否则短文档配额用不满
                continue
            if size >= quota * 0.85 and size > 0:
                break
            picked.append((i, s))
            size += len(s)

    # 按原文顺序还原，段落顺序本身也是信息（"一、回购方案 / 二、实施进展"）
    picked.sort(key=lambda x: x[0])
    out, size = [], 0
    for _, s in picked:
        if size + len(s) > max_chars * 1.1:
            break
        out.append(s)
        size += len(s)
    return "\n".join(out)


# ---------------------------------------------------------------- 写入
def save_titles(rows: list[dict]) -> int:
    if not rows:
        return 0
    con = _con()
    try:
        con.executemany(
            "INSERT OR REPLACE INTO title (art_code, ts_code, name, title, ntype, trade_date, url) "
            "VALUES (?,?,?,?,?,?,?)",
            [(r["art_code"], r.get("ts_code"), r.get("name"), r["title"],
              r.get("ntype"), r.get("trade_date"), r.get("url")) for r in rows],
        )
        con.commit()
    finally:
        con.close()
    return len(rows)


def budget_for(n_chars: int) -> int:
    """关键段预算按原文长度阶梯放大。

    固定 700 字符对 3000 字的日常公告是 23% 覆盖，够用；
    对 19 万字的问询函回复是 0.4%，等于随机摘一句。
    超长公告本来就少（P0 里不到 5%），多给 2 倍预算对总量影响可忽略。
    """
    if n_chars > 50000:
        return 1500
    if n_chars > 5000:
        return 950
    return KEY_CHARS


def save_body(art_code: str, row: dict, text: str, n_pages: int,
              title: str = "") -> int:
    """落库。返回正文字符数（0 表示没拿到文本）。

    ## ok 三态，别把"没文本"和"没抓到"混一起

    `ok=1` 有正文；`ok=0` 一条都没拿到（网络/接口异常，值得重试）；
    `ok=2` 接口通了、也确实有这条公告，但正文是纯 PDF 附件、文本层为空
    （重组报告书、审计报告常见，例如"创远信科:国泰海通证券…发行股份及支付现金购买资产"）。

    三者混成 0 的后果实测过：每天补数都会重试那两条 PDF 公告，
    永远失败，日志里天天两行红字，真正的网络失败反而被淹没。
    """
    con = _con()
    try:
        if text:
            ok = 1
        elif n_pages >= 1:
            ok = 2          # 接口有响应页，只是没有文本层
        else:
            ok = 0
        con.execute(
            "INSERT OR REPLACE INTO notice (art_code, ts_code, name, title, ntype, trade_date, "
            "url, prio, n_pages, raw_chars, key_text, body, fetched_at, ok) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [art_code, row.get("ts_code"), row.get("name"), title or row.get("title"),
             row.get("ntype"), row.get("trade_date"), row.get("url"),
             priority(row.get("ntype") or "", row.get("title") or ""),
             n_pages, len(text), key_paragraphs(text, budget_for(len(text))) if text else "",
             zlib.compress(text.encode("utf-8"), 6) if text else b"",
             time.strftime("%Y-%m-%d %H:%M:%S"), ok],
        )
        con.commit()
    finally:
        con.close()
    return len(text)


def harvest(date: str, prio_max: str = "P0", workers: int = 3,
            limit: int | None = None, verbose: bool = True) -> dict:
    """抓某天标题，并按类型分级抓正文。

    prio_max: "P0" 只抓白名单；"P1" 连 P1 一起；"P2" 全抓（不建议）。

    ## 撞上风控就整体停，不要在单条上重试

    第一版遇到 567 直接算"失败"继续下一条，结果 629 条里 556 条失败——
    因为封禁是按 IP 的，重试只是把封禁时间拖长。现在的策略是：
    任一线程拿到 567 → 全局暂停 30 秒（`_backoff`）→ 连续 12 次限流就整批中止，
    把已经抓到的存下来，等下一轮再补。增量抓取本来就是幂等的，
    半途放弃比拖到被封一小时好。
    """
    t0 = time.time()
    rows = fetch_titles(date)
    save_titles(rows)
    want = {"P0": ("P0",), "P1": ("P0", "P1"), "P2": ("P0", "P1", "P2")}[prio_max]
    todo = [r for r in rows if priority(r["ntype"], r["title"]) in want]
    # 已抓过的不重抓：harvest 会被反复调用（每天增量、补漏、重建索引），
    # 不跳过就得为同一批公告重复付网络时间。ok=0 的空公告也算抓过，避免死循环重试。
    con = _con()
    try:
        have = {r[0] for r in con.execute("SELECT art_code FROM notice").fetchall()}
    finally:
        con.close()
    before = len(todo)
    todo = [r for r in todo if r["art_code"] not in have]
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"  {date}: 标题 {len(rows)} 条，需抓正文 {len(todo)} 条"
              f"（P0 {sum(1 for r in rows if priority(r['ntype'], r['title']) == 'P0')}"
              f"，已抓过 {before - len(todo)}）")

    done = {"成功": 0, "空": 0, "失败": 0, "限流": 0, "字符": 0}
    sess = _session()
    stopped = {"hit": 0, "abort": False}
    lock = __import__("threading").Lock()

    def one(r):
        if stopped["abort"]:
            return ("失败", 0)
        try:
            d = fetch_body(r["art_code"], sess=sess)
            n = save_body(r["art_code"], r, d["正文"], d["页数"], d["标题"])
            with lock:
                stopped["hit"] = 0
            return ("成功", n) if n else ("空", 0)
        except RateLimited:
            with lock:
                stopped["hit"] += 1
                if stopped["hit"] >= 12:
                    stopped["abort"] = True
            _backoff(30)
            return ("限流", 0)
        except Exception:
            return ("失败", 0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for st, n in pool.map(one, todo):
            done[st] += 1
            done["字符"] += n
    elapsed = time.time() - t0
    if verbose:
        print(f"    抓取 {done['成功']} 成功 / {done['空']} 空 / {done['失败']} 失败"
              f"{' / ' + str(done['限流']) + ' 限流' if done['限流'] else ''}，"
              f"{done['字符'] / 1e4:.1f} 万字符，{elapsed:.0f}s")
        if stopped["abort"]:
            print("    ⚠ 连续限流，已中止本批。等 5-10 分钟重跑同样的命令即可续上（幂等）。")
    return {"日期": date, "标题": len(rows), "抓正文": len(todo), **done,
            "耗时": round(elapsed, 1)}


# ---------------------------------------------------------------- 读取
def get(art_code: str, full: bool = False) -> dict | None:
    """先查本地，没有再抓（抓到就存）。full=True 返回原文，否则关键段落。"""
    con = _con()
    try:
        r = con.execute("SELECT * FROM notice WHERE art_code = ?", [art_code]).fetchone()
    finally:
        con.close()
    if r:
        return {
            "art_code": r["art_code"], "title": r["title"], "ts_code": r["ts_code"],
            "name": r["name"], "ntype": r["ntype"], "trade_date": r["trade_date"],
            "url": r["url"], "raw_chars": r["raw_chars"], "n_pages": r["n_pages"],
            "正文": zlib.decompress(r["body"]).decode("utf-8") if (full and r["body"]) else None,
            "关键段落": r["key_text"], "来自": "本地",
        }
    # 本地没有：拿标题层的信息补上下文，再抓
    con = _con()
    try:
        t = con.execute("SELECT * FROM title WHERE art_code = ?", [art_code]).fetchone()
    finally:
        con.close()
    row = dict(t) if t else {"art_code": art_code, "title": "", "ntype": "",
                             "trade_date": "", "url": ""}
    d = fetch_body(art_code)
    if not d["正文"]:
        return None
    save_body(art_code, row, d["正文"], d["页数"], d["标题"])
    return {
        "art_code": art_code, "title": d["标题"] or row.get("title"),
        "ts_code": row.get("ts_code"), "name": row.get("name"),
        "ntype": row.get("ntype"), "trade_date": row.get("trade_date"),
        "url": row.get("url"), "raw_chars": len(d["正文"]), "n_pages": d["页数"],
        "正文": d["正文"] if full else None,
        "关键段落": key_paragraphs(d["正文"]), "来自": "网络",
    }


def stats() -> dict:
    con = _con()
    try:
        t = con.execute("SELECT count(*) c FROM title").fetchone()["c"]
        b = con.execute("SELECT count(*) c, sum(raw_chars) s, sum(length(key_text)) k "
                        "FROM notice WHERE ok=1").fetchone()
        by = con.execute("SELECT prio, count(*) c FROM notice GROUP BY prio").fetchall()
        pdf = con.execute("SELECT count(*) c FROM notice WHERE ok=2").fetchone()["c"]
        size = DB.stat().st_size if DB.exists() else 0
        # 覆盖度：按日算 P0 标题有多少还没抓到正文。盘中公告一直在涨，
        # 这个数不会归零，只要能在收盘后落到 0 就说明当天补全了。
        rows = con.execute("SELECT t.art_code, t.ntype, t.title, t.trade_date FROM title t").fetchall()
        have = {r[0] for r in con.execute("SELECT art_code FROM notice").fetchall()}
        per: dict[str, dict] = {}
        for a, nt, ti, d in rows:
            if priority(nt or "", ti or "") != "P0":
                continue
            k = per.setdefault(d, {"P0": 0, "待抓": 0})
            k["P0"] += 1
            if a not in have:
                k["待抓"] += 1
        p0 = sum(v["P0"] for v in per.values())
        left = sum(v["待抓"] for v in per.values())
        return {"标题条数": t, "有正文": b["c"], "正文字符": b["s"] or 0,
                "关键段字符": b["k"] or 0,
                "压缩比": round((b["k"] or 0) / (b["s"] or 1), 3),
                "分级": {r["prio"]: r["c"] for r in by},
                "无文本PDF": pdf,
                "P0覆盖": {"标题": p0, "已抓": p0 - left, "待抓": left,
                          "完成率": round((p0 - left) / p0, 3) if p0 else 1.0,
                          "按日": {d: v for d, v in sorted(per.items())}},
                "库大小MB": round(size / 1048576, 2)}
    finally:
        con.close()


# ---------------------------------------------------------------- 接 RAG
def build_index(days: int | None = None, only_missing: bool = True,
                verbose: bool = True) -> dict:
    """把关键段落灌进 rag.db，kind='notice_body'。

    ## 为什么单独一个 kind 而不是覆盖原来的 notice 标题块

    标题块回答"谁发了什么"，正文块回答"具体条款是什么"，是两个问题。
    混成一个 kind 会让"最近有哪些减持公告"这种查询被正文块淹没——
    正文块的词项密度高得多，BM25 分数天然占优，标题块会被压到看不见。

    分 kind 之后，模型可以按需选 `kind='notice'`（列公告）或
    `kind='notice_body'`（查条款），也可以不指定让两层一起出。
    """
    from src.agent import rag as R

    # 门槛 24 字：正文抽出来的关键段落再短就没信息量，但 40-80 字的专项说明
    # （"同意变更签字律师"）本身就只有一句话，一刀切 80 会把真公告挡在外面。
    MIN_KEY = 24
    con = _con()
    try:
        sql = "SELECT art_code, ts_code, name, title, ntype, trade_date, url, key_text, ok " \
              f"FROM notice WHERE (ok=1 AND length(key_text) > {MIN_KEY}) OR ok=2"
        rows = con.execute(sql).fetchall()
        alive = {f"nbody::{r['art_code']}" for r in rows}
    finally:
        con.close()
    if days:
        # days 传进来是"日期下限"（如 20261101），只控制本次索引范围，不影响清理范围
        rows = [r for r in rows if (r["trade_date"] or "") >= str(days)]

    if only_missing:
        # 已经索引过的不重复提交：块数上万后，全量重提交会白跑几分钟
        st = R.stats()
        if st.get("分类", {}).get("notice_body"):
            done = _indexed_arts()
            rows = [r for r in rows if r["art_code"] not in done]

    chunks = []
    for r in rows:
        if r["ok"] == 2:
            # PDF-only 公告：不给空壳块，留一条可解释的占位，
            # 这样"这家公司重组进展如何"能命中并说明"正文需看原页"，而不是命中 0 条
            body = (f"【{r['title']}】\n该公告正文为纯 PDF 附件，东财正文接口没有文本层，"
                    f"需打开原页查看：{r['url'] or '东财公告详情页'}")
        else:
            body = f"【{r['title']}】\n{r['key_text']}"
        chunks.append({
            "doc_key": f"nbody::{r['art_code']}",
            "source": r["name"] or "公告",
            "kind": "notice_body",
            "title": r["title"],
            "text": body,
            "ts_code": r["ts_code"],
            "trade_date": r["trade_date"],
            "url": r["url"],
        })
    n = R.upsert_chunks(chunks, verbose=False) if chunks else 0
    # 正文表里 ok=0 / 被删的公告，索引里对应块也要清掉，否则检索会命中空壳
    pr = R.prune_keys("notice_body", alive)
    if chunks:
        R.mark_built()
    if verbose:
        print(f"[公告正文] 提交 {len(chunks)} 条，新增 {n} 块，清理 {pr['删除']} 块")
    return {"提交": len(chunks), "新增": n, "清理": pr["删除"]}


def _indexed_arts() -> set[str]:
    con = _con()
    try:
        con.execute("ATTACH DATABASE ? AS r", [str(DATA_DIR / "rag.db")])
        rows = con.execute(
            "SELECT doc_key FROM r.chunk WHERE doc_key LIKE 'nbody::%'").fetchall()
        return {r["doc_key"].split("::", 1)[1] for r in rows}
    except Exception:
        return set()
    finally:
        con.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        print(json.dumps(stats(), ensure_ascii=False, indent=2))
    else:
        d = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d")
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print(json.dumps(harvest(d, limit=lim), ensure_ascii=False))
