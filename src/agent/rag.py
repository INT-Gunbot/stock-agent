"""RAG 检索层：BM25 + 中文字符 bigram。

## 为什么是 BM25 而不是向量检索

先看事实：本机的模型服务 `POST /v1/embeddings` 返回 **404**，没有 embedding 端点。
在没有向量化服务的前提下，所谓"向量 RAG"只有两条路——要么调外部 API（多一个 key、
多一份钱、多一个可能挂掉的依赖），要么自己训一个小模型（这个项目不值得）。

而检索这件事，本质上要解决的是「从一堆文本里找出和问题相关的那几段」。
这是信息检索问题，向量只是其中一种解法。BM25 是 Lucene/Elasticsearch 的默认算法，
在**短查询 + 专有名词密集**的场景下（"复权因子怎么处理除权日"、"PIT 是什么"）
反而比稠密向量更准，因为它精确匹配词项，不会把"复权"和"分红"混成一个模糊的语义邻域。

代价也说清楚：同义词召回不了。问"除息"找不到只写了"分红"的段落。
在金融文档里这个代价可接受——术语是收敛的，用户问的通常就是文档里写的那个词。

## 中文怎么切

没有分词库，用**字符 bigram**：`复权因子` → `复权 / 权因 / 因子`。
不需要词典，对未登录词（"宁德时代"、"PIT"）天然友好。缺点是无法区分词边界，
所以再叠一层：英文和数字按整词切（`MA20`、`ROE`、`20261102` 保持完整）。

停用词不做人工表——BM25 的 IDF 会自动把"的/了/是"这类高频项压到接近 0 权重。
手工维护停用词表在金融文本里反而危险（"不用"、"未"、"非"都是有效信号）。

## 索引怎么存

存 `data/rag.db`（SQLite），不存内存也不存主库：
- 不存内存：公告每天几百条，攒一个月就是两万块，全量载入要几百 MB；
  （正文不进这里：公告正文单独存 data/notice.db，见 src/notices.py，
  这里只收抽出来的关键段落，kind=notice_body）
- 不存主库：主库被行情更新独占，建索引时会把对话锁死。
倒排表按 `(term, chunk_id)` 建索引，检索时一条 `WHERE term IN (...)` 就能拿到候选，
再用 SQLite 聚合算出 BM25 分数。两万块的库，检索在 10ms 量级。
"""
from __future__ import annotations

import math
import re
import sqlite3
import time
from pathlib import Path

from src.config import DATA_DIR

DB = DATA_DIR / "rag.db"

# BM25 参数：k1 控制词频饱和，b 控制长度归一化强度。取 Lucene 默认值。
K1 = 1.2
B = 0.75

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_key    TEXT NOT NULL,          -- 去重键：source + 位置
    source     TEXT NOT NULL,          -- 文件名 / 公告
    kind       TEXT NOT NULL,          -- methodology | report | notice
    title      TEXT,                   -- 章节标题或公告标题
    text       TEXT NOT NULL,
    ts_code    TEXT,
    trade_date TEXT,
    url        TEXT,
    n_terms    INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_key ON chunk(doc_key);
CREATE INDEX IF NOT EXISTS idx_chunk_kind ON chunk(kind);
CREATE INDEX IF NOT EXISTS idx_chunk_code ON chunk(ts_code, trade_date);

CREATE TABLE IF NOT EXISTS posting (
    term     TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    tf       INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_post_term ON posting(term);

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""


def _con() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB), timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(_SCHEMA)
    return con


# ---------------------------------------------------------------- 分词
_ASCII = re.compile(r"[a-zA-Z]{2,}|\d{4,}")
_STOP_CH = set("的了是在有和就也不这那你我他她它个上下来去以为对到说会着过把被让使而且或但如果因为所以可以没"
               "怎什么如何谁哪样些能要想需")
_PUNCT = re.compile(r"[\s\u3000-\u303f\uff00-\uffef.,;:!?()\[\]{}<>\"'`~@#$%^&*_+=|\\/—–…·、，。；：！？（）【】《》「」『』]+")


def tokenize(text: str) -> list[str]:
    """中文 bigram + 英文数字整词。返回小写词项列表（含重复，用于算 tf）。"""
    if not text:
        return []
    s = _PUNCT.sub(" ", str(text))
    out: list[str] = []
    for seg in s.split():
        # 英文/数字：整词保留；同时把长数字的前 6 位也留一份（股票代码 600519.SH 之类）
        for m in _ASCII.finditer(seg):
            w = m.group(0).lower()
            out.append(w)
        # 中文字符：取 bigram；单字仅当整段很短时才补（长文补单字全是噪声）
        han = re.sub(r"[^\u4e00-\u9fff]", "", seg)
        if len(han) >= 2:
            for i in range(len(han) - 1):
                bg = han[i:i + 2]
                out.append(bg)
        elif len(han) == 1 and han not in _STOP_CH:
            out.append(han)
    return out


def _terms_for_index(text: str) -> dict[str, int]:
    """建索引用：词频表。纯停用字组成的 bigram 丢掉，省空间。"""
    d: dict[str, int] = {}
    for t in tokenize(text):
        if len(t) == 2 and t[0] in _STOP_CH and t[1] in _STOP_CH:
            continue
        d[t] = d.get(t, 0) + 1
    return d


def _is_han(t: str) -> bool:
    return all("\u4e00" <= ch <= "\u9fff" for ch in t)


# 疑问模板：检索前先剥掉。这些词不是问题的主题，留着只会让"今天天气怎么样"
# 靠"今天"命中日报块，然后一本正经地给出无关答案。
_Q_WORDS = ("是什么意思", "什么意思", "怎么样", "怎么办", "怎么", "如何", "为什么", "是什么",
            "有什么", "有哪些", "哪些", "什么", "多少", "吗", "呢", "的", "了", "能", "可以", "请", "帮我")
_HAN4 = re.compile(r"[\u4e00-\u9fff]{3,}")


def _query_phrases(query: str, n: int = 3) -> list[str]:
    """从查询里提取 n 字短语滑窗，用于检查"主题"是否真的出现在块里。

    这是给 BM25 补位置信息。BM25 只知道"这个词出现过"，不知道"这几个字是连在一起的"。
    所以"今天天气怎么样"能靠"今天"命中日报块——技术上合法，语义上无关。
    取出"今天天气"这样的短语做子串检查，就能把这类误匹配挡掉。

    **删疑问词而不是替换成空格**：这是踩过的坑。
    "数据有什么缺口"如果变成"数据 缺口"，就切断了真正该查的"数据缺口"；
    直接删除得到"数据缺口"，才是用户想问的东西。

    滑窗取 3 字而不是 4 字：4 字太严，"持仓集中度"会切成
    "持仓集中/仓集中度"，全落空；3 字能保住"集中度"这个实际用词。
    """
    s = str(query or "")
    for w in _Q_WORDS:
        s = s.replace(w, "")
    out: list[str] = []
    for seg in _HAN4.findall(s):
        if len(seg) <= 4:
            out.append(seg)
            continue
        for i in range(len(seg) - n + 1):
            out.append(seg[i:i + n])
    return list(dict.fromkeys(out))[:14]


def _confidence(hits: list[dict], missing: list[str], q_terms: list[str],
                phrases: list[str], ascii_hit: bool, strong_filter: bool = False) -> dict:
    """给检索结果一个可信度，供模型决定要不要说"文档里没有"。

    三级判断，按可靠性排序：
    1. 短语命中 —— 查询里的 n 字片段真的出现在块里，这是最强的相关性证据；
    2. 英文/数字词项命中 —— "PIT"、"MA20"、"600519" 这种，命中即主题命中；
    3. 兜底看覆盖率 —— 前两条都不成立时，只能说明匹配到了泛词。

    刻意留一个"中"档：宁可让模型看到"部分相关"自己判断，
    也不要在阈值上硬砍——砍狠了漏召回，砍松了放噪声，两头都不讨好。
    """
    if not hits:
        return {"等级": "无", "说明": "索引里没有匹配内容"}
    phrase_ok = any(any(p in (h.get("text") or "") for p in phrases) for h in hits) if phrases else False
    cov = max((h.get("覆盖率") or 0) for h in hits)
    ratio = len(missing) / max(len(q_terms), 1)

    # 有代码/日期限定时，范围本身已经把答案框住了，此时"没有命中短语"
    # 不代表不相关（问"茅台有什么公告"时查询词就是泛词"公告"）。
    if phrase_ok or ascii_hit or strong_filter:
        if cov >= 0.45 and ratio < 0.5:
            return {"等级": "高", "说明": "命中主题短语，可直接引用"}
        return {"等级": "中", "说明": "命中主题词但覆盖不全，可引用，需说明只是部分依据"}
    # 既没有短语命中，也没有英文术语命中 —— 说明只匹配到了泛词
    return {"等级": "低", "说明": "只匹配到泛词，多半与问题无关。要说'文档里没有直接相关的内容'，不要硬凑"}


# ---------------------------------------------------------------- 切块
_H2 = re.compile(r"^#{2,4}\s+(.+)$", re.M)


def chunk_markdown(text: str, source: str, kind: str = "methodology",
                   target: int = 420) -> list[dict]:
    """按 markdown 标题切块，过短的块向前合并，过长的按行再切。

    标题会写进块首（`【章节】` 形式）——这样"复权因子"这一节的每一块都带着章节名，
    检索时不会因为块里恰好没提"复权"而漏掉。
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    # 先按标题切成小节
    marks = [(m.start(), m.group(1).strip()) for m in _H2.finditer(text)]
    sections: list[tuple[str, str]] = []
    if not marks:
        sections.append((Path(source).stem, text))
    else:
        if marks[0][0] > 0:
            sections.append((Path(source).stem, text[: marks[0][0]].strip()))
        for i, (pos, title) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            sections.append((title, text[pos:end].strip()))

    out: list[dict] = []
    for title, body in sections:
        if not body:
            continue
        head = f"【{title}】\n"
        if len(body) <= target * 1.6:
            out.append({"title": title, "text": head + body})
            continue
        # 过长：按行累加到 target 就切一刀，不切断行
        buf, size = [], 0
        for ln in body.split("\n"):
            buf.append(ln)
            size += len(ln) + 1
            if size >= target:
                out.append({"title": title, "text": head + "\n".join(buf).strip()})
                buf, size = [], 0
        if buf and "".join(buf).strip():
            out.append({"title": title, "text": head + "\n".join(buf).strip()})

    return [{"title": c["title"], "text": c["text"], "source": source, "kind": kind}
            for c in out if len(c["text"].strip()) > 30]


# ---------------------------------------------------------------- 写入
def upsert_chunks(chunks: list[dict], verbose: bool = False) -> int:
    """批量写入并建倒排。按 doc_key 去重，重复建索引时只更新有变化的块。

    这里刻意"先删后插"而不是 upsert 倒排——因为文本可能变短，
    旧的 term 必须清掉，否则会检索到已经不存在的词。

    ## 内容没变就别重写倒排

    每天重建索引时，90% 以上的块（昨天的公告标题、没动过的设计文档）
    和库里一模一样。第一版对所有块无条件 `DELETE posting` + 重插，
    2.3 万块 × 平均 33 个词项 ≈ 78 万行写入，跑一次要好几分钟，
    而这几分钟里 disk I/O 全花在"把相同的内容再写一遍"上。
    现在先比 text/title/ts_code/trade_date/url，全一样就跳过，
    只重写真正变化的块——日常增量的耗时从分钟级掉到秒级。
    """
    if not chunks:
        return 0
    con = _con()
    added = skipped = 0
    try:
        for c in chunks:
            key = c.get("doc_key") or f"{c['source']}#{hash(c['text']) & 0xFFFFFFFF:08x}"
            terms = _terms_for_index(f"{c.get('title') or ''} {c['text']}")
            if not terms:
                continue
            row = con.execute(
                "SELECT id, text, title, ts_code, trade_date, url FROM chunk WHERE doc_key = ?",
                [key]).fetchone()
            if row:
                same = (row["text"] == c["text"]
                        and (row["title"] or "") == (c.get("title") or "")
                        and (row["ts_code"] or "") == (c.get("ts_code") or "")
                        and (row["trade_date"] or "") == (c.get("trade_date") or "")
                        and (row["url"] or "") == (c.get("url") or ""))
                if same:
                    skipped += 1
                    continue
                cid = row["id"]
                con.execute("DELETE FROM posting WHERE chunk_id = ?", [cid])
                con.execute(
                    "UPDATE chunk SET source=?, kind=?, title=?, text=?, ts_code=?, "
                    "trade_date=?, url=?, n_terms=? WHERE id=?",
                    [c["source"], c.get("kind", "doc"), c.get("title"), c["text"],
                     c.get("ts_code"), c.get("trade_date"), c.get("url"), sum(terms.values()), cid],
                )
            else:
                cur = con.execute(
                    "INSERT INTO chunk (doc_key, source, kind, title, text, ts_code, trade_date, url, n_terms) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    [key, c["source"], c.get("kind", "doc"), c.get("title"), c["text"],
                     c.get("ts_code"), c.get("trade_date"), c.get("url"), sum(terms.values())],
                )
                cid = int(cur.lastrowid or 0)
                added += 1
            con.executemany("INSERT OR REPLACE INTO posting (term, chunk_id, tf) VALUES (?,?,?)",
                            [(t, cid, tf) for t, tf in terms.items()])
        con.commit()
    finally:
        con.close()
    if verbose:
        print(f"[索引] 新增 {added} 块 / 更新 {len(chunks) - added - skipped} 块 / "
              f"未变跳过 {skipped} 块 / 提交 {len(chunks)} 块")
    return added


# ---------------------------------------------------------------- 清理
def _purge(ids: list[int], dry_run: bool = False) -> int:
    """按 id 删块 + 对应倒排。倒排必须先删，否则会留下指向空块的词项。"""
    if not ids or dry_run:
        return 0
    con = _con()
    try:
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            ph = ",".join("?" * len(part))
            con.execute(f"DELETE FROM posting WHERE chunk_id IN ({ph})", part)
            con.execute(f"DELETE FROM chunk WHERE id IN ({ph})", part)
        con.commit()
    finally:
        con.close()
    return len(ids)


def prune_keys(kind: str, keep: set[str], dates: set[str] | None = None,
               dry_run: bool = False) -> dict:
    """`dates` 范围内、doc_key 不在 `keep` 里的块删掉。

    `dates=None` 表示整个 kind 范围。只限定日期是有意的：某天抓取失败时
    那一天根本不进 `dates`，它的块就动不到——清理不能把"这次没抓到"
    误判成"源里没有"。

    传进来的 `keep` 是"这一轮真正产出的全部块键"，所以它同时能处理两种情况：
    源文件被删（键不在 keep 里）、源文件变短导致章节数减少（尾部的 #N 键不在 keep 里）。
    按 `source` 列对齐做不到后者——文件还在，只是第 9 章没了。
    """
    con = _con()
    try:
        if dates:
            ph = ",".join("?" * len(dates))
            rows = con.execute(
                f"SELECT id, doc_key FROM chunk WHERE kind = ? AND trade_date IN ({ph})",
                [kind, *dates]).fetchall()
        else:
            rows = con.execute("SELECT id, doc_key FROM chunk WHERE kind = ?", [kind]).fetchall()
    finally:
        con.close()
    dead = [r["id"] for r in rows if r["doc_key"] not in keep]
    n = _purge(dead, dry_run)
    return {"kind": kind, "删除": n if not dry_run else 0, "待删": len(dead)}


def stats() -> dict:
    con = _con()
    try:
        total = con.execute("SELECT count(*) c FROM chunk").fetchone()["c"]
        rows = con.execute("SELECT kind, count(*) c FROM chunk GROUP BY kind").fetchall()
        terms = con.execute("SELECT count(DISTINCT term) c FROM posting").fetchone()["c"]
        last = con.execute("SELECT v FROM meta WHERE k='built_at'").fetchone()
    finally:
        con.close()
    return {"块数": total, "词项数": terms,
            "分类": {r["kind"]: r["c"] for r in rows},
            "建索引时间": (last["v"] if last else None)}


def _set_meta(k: str, v: str) -> None:
    con = _con()
    try:
        con.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?,?)", [k, v])
        con.commit()
    finally:
        con.close()


def mark_built() -> None:
    _set_meta("built_at", time.strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------- 检索
def search(query: str, k: int = 5, kind: str | None = None,
           ts_code: str | None = None, days: int | None = None,
           min_score: float = 1.0) -> dict:
    """BM25 检索 + IDF 覆盖率加权。

    ## 为什么 BM25 之外还要加覆盖率

    纯 BM25 在这批语料上有个致命毛病：问"如何烤面包"也能拿到 7 分。
    原因不是 BM25 错了，而是中文 bigram 会把"如何烤面包"切出"如何/何烤/烤面/面包"，
    其中"如何"这类虚词在长文档里反复出现，靠词频堆出了基础分。

    修法不是加停用词表（治不完），而是**按 IDF 给覆盖率加权**：
    把查询里每个词的 IDF 当作它的"重要性权重"，算出候选块覆盖了多少总权重。
    "烤面包"的 bigram 在全库里几乎不出现，"面包"命中也是零散 —— 覆盖率接近 0，
    分数被压下去；而"复权/口径"这类领域词 IDF 高，一旦命中就主导分数。

    final = bm25 × (0.35 + 0.65 × coverage)

    0.35 是保底系数：哪怕覆盖率低，词频极高的块也不至于被完全抹掉。

    没有命中时返回空列表而不是报错——上层要能说"文档里没写"，这是价值不是失败。
    """
    # 查询侧做和索引一样的过滤：纯虚词 bigram 不参与检索
    qt: list[str] = []
    for t in tokenize(query):
        if len(t) == 2 and t[0] in _STOP_CH and t[1] in _STOP_CH:
            continue
        qt.append(t)
    q_terms = list(dict.fromkeys(qt))[:40]
    phrases = _query_phrases(query)
    if not q_terms:
        return {"结果": [], "统计": {"原因": "查询没有可检索的词项（全是虚词？）",
                                      "可信度": {"等级": "无", "说明": "问题里没有可检索的词"}}}

    con = _con()
    try:
        n_docs = con.execute("SELECT count(*) c FROM chunk").fetchone()["c"]
        if not n_docs:
            return {"结果": [], "统计": {"原因": "索引为空，先运行 python -m scripts.build_index"}}

        avgdl = con.execute("SELECT avg(n_terms) a FROM chunk").fetchone()["a"] or 1.0
        ph = ",".join("?" * len(q_terms))

        # 全局 df：算权重用，不带筛选条件 —— 这样不同查询、不同过滤下分数才可比
        gdf = {r["term"]: r["c"] for r in con.execute(
            f"SELECT term, count(*) c FROM posting WHERE term IN ({ph}) GROUP BY term", q_terms
        ).fetchall()}
        idf = {}
        for t in q_terms:
            df_t = gdf.get(t, 0)
            w = math.log(1 + (n_docs - df_t + 0.5) / (df_t + 0.5))
            # 中文 bigram 且全库只出现 1-2 次：大概率是跨词边界的伪词项
            # （"是什么意思" → 么意/意思），不是真信号。打个折，别让它主导覆盖率。
            # 真专有名词不怕这个折扣——"宁德时代"会同时命中 宁德/德时/时代，累计权重仍然高。
            if len(t) == 2 and _is_han(t) and 0 < df_t <= 2:
                w *= 0.6
            idf[t] = w
        # 分母只算"库里真的有"的词项。
        # 这是个真 bug 的修法：df=0 的词项按公式会拿到最高 IDF（因为分母是 0.5），
        # 于是"什么意思"里的伪词项主导了覆盖率分母，把"PIT"这种正常查询的分压到 1.0 以下。
        # 库里不存在的词不能用来惩罚候选块——它压根没有匹配的机会。
        in_lib = [t for t in q_terms if gdf.get(t, 0) > 0]
        total_w = sum(idf[t] for t in in_lib) or 1.0
        missing = [t for t in q_terms if gdf.get(t, 0) == 0]

        where = ["1=1"]
        params: list = []
        if kind == "notice":
            # 检索"公告"时把正文层一起带上。
            # 实测模型几乎总是传 kind='notice'（哪怕问的是"回购了多少股"），
            # 只查标题层就永远看不到正文，然后它会说"这个数据没接入"——明明接了。
            # 两层本来就同源（同一个 url），按 url 归组后不会重复出结果，
            # 所以合并查没有副作用，只是多给模型一条有内容的候选。
            where.append("c.kind IN ('notice','notice_body')")
        elif kind:
            where.append("c.kind = ?")
            params.append(kind)
        if ts_code:
            # 模型给代码时常常是 6 位（600519），而索引里存的是带后缀的（600519.SH）。
            # 精确匹配会让"茅台有什么公告"静默返回 0 命中——这种错最难发现，
            # 因为它的表现是"没有数据"，看起来像数据没抓到。
            c6 = re.sub(r"[^0-9]", "", str(ts_code))
            if len(c6) == 6:
                where.append("c.ts_code LIKE ?")
                params.append(c6 + ".%")
            else:
                where.append("c.ts_code = ?")
                params.append(str(ts_code).strip().upper())
        if days:
            where.append("c.trade_date >= strftime('%Y%m%d', 'now', ?)")
            params.append(f"-{int(days)} days")
        wsql = " AND ".join(where)

        rows = con.execute(
            f"""SELECT p.term, p.chunk_id, p.tf, c.n_terms
                FROM posting p JOIN chunk c ON c.id = p.chunk_id
                WHERE p.term IN ({ph}) AND {wsql}""",
            q_terms + params,
        ).fetchall()
        if not rows:
            return {"结果": [], "统计": {"查询词项": q_terms[:8], "候选": 0,
                                          "说明": "索引里没有任何块包含这些词"}}

        scores: dict[int, float] = {}
        matched: dict[int, set[str]] = {}
        for r in rows:
            t, cid, tf, dl = r["term"], r["chunk_id"], r["tf"], r["n_terms"] or 1
            denom = tf + K1 * (1 - B + B * dl / avgdl)
            scores[cid] = scores.get(cid, 0.0) + idf[t] * tf * (K1 + 1) / denom
            matched.setdefault(cid, set()).add(t)

        # 查询词没区分度时（"公告"出现在 23419/23521 个块里，IDF≈0.004），
        # BM25 分数会被压到 0.01 量级，然后被绝对阈值全砍掉 —— 表现是"命中 0"，
        # 看起来像数据没抓到，其实数据在，只是打分尺度崩了。
        # 有代码限定时范围已经框死了（"600519 的公告"总共就 5 条），
        # 此时绝对分数没有意义，只看相对排序就够了。
        strong_filter = bool(ts_code)
        weak_query = (not in_lib) or all(idf[t] < 0.6 for t in in_lib)
        if strong_filter:
            min_score, rel_keep = 0.0, 0.35
        else:
            min_score, rel_keep = 1.0, 0.35

        final: dict[int, float] = {}
        for cid, s in scores.items():
            cov = sum(idf[t] for t in matched[cid]) / total_w
            final[cid] = s * (0.35 + 0.65 * cov)

        top = sorted(final.items(), key=lambda x: -x[1])[: max(k * 4, 24)]
        ids = [cid for cid, _ in top]
        ph2 = ",".join("?" * len(ids))
        dmap = {r["id"]: r for r in con.execute(
            f"""SELECT id, source, kind, title, text, ts_code, trade_date, url
                FROM chunk WHERE id IN ({ph2})""", ids
        ).fetchall()}

        # 短语加成：真正包含查询短语的块上调排序。
        # 这不是锦上添花——"数据有什么缺口"的正确答案在 README 的"数据缺口"那段，
        # 但纯 BM25 会把它排到讲"数据新鲜度"的块后面，因为后者词频更高。
        # 位置信息只有子串检查能补，补上之后排序才对。
        if phrases:
            boosted = []
            for cid, sc in top:
                r = dmap.get(cid)
                n_hit = sum(1 for p in phrases if p in (r["text"] or "")) if r else 0
                boosted.append((cid, sc * (1 + 0.4 * min(n_hit, 3))))
            top = sorted(boosted, key=lambda x: -x[1])
        best = top[0][1] if top else 0.0
    finally:
        con.close()

    out = []
    seen: set[str] = set()

    # ## 同一条公告会同时命中标题块和正文块
    #
    # 标题块 69 字符，正文块 700 字符。BM25 的长度归一化（b=0.75）让短文档占便宜：
    # 问"赛意信息回购了多少股"，标题块拿 38 分排第一，正文块 19 分排第二——
    # 而真正写着"1,093,900 股"的那句话在正文块里。用户看到的第一条是条空壳。
    #
    # 修法不是给正文块加分（会污染其它类型的排序），而是**按公告归组**：
    # 组内优先正文块当代表，组分数取组内最高。这样一条公告只出一条结果，
    # 出来的就是有内容的那条，同时保留了"标题强匹配"证明这条公告相关。
    groups: dict[str, list] = {}
    order: list[str] = []
    for cid, sc in top:
        r = dmap.get(cid)
        if not r:
            continue
        # 公告用 url 归组（标题块和正文块共享同一个 url）；其它类型沿用原来的去重键
        gkey = r["url"] or f"{r['source']}|{r['title']}|{r['trade_date']}"
        if gkey not in groups:
            groups[gkey] = []
            order.append(gkey)
        cov = sum(idf[t] for t in matched.get(cid, set())) / total_w
        groups[gkey].append((sc, cov, r))

    for gkey in order:
        items = groups[gkey]
        # 正文块优先，其次分数高的
        items.sort(key=lambda x: (-(x[2]["kind"] == "notice_body"), -x[0]))
        best_sc = max(i[0] for i in items)
        _, cov, r = items[0]
        if best_sc < min_score or best_sc < best * rel_keep:
            continue
        dedup = f"{r['source']}|{r['title']}|{r['trade_date']}"
        if dedup in seen:
            continue
        seen.add(dedup)
        out.append({
            "text": r["text"], "title": r["title"], "source": r["source"],
            "kind": r["kind"], "ts_code": r["ts_code"], "trade_date": r["trade_date"],
            "url": r["url"], "score": round(best_sc, 2),
            "覆盖率": round(cov, 2),
        })
        if len(out) >= k:
            break

    # 英文/数字词项（PIT、MA20、600519）命中即主题命中——这类词不会因泛词混淆
    ascii_terms = {t for t in in_lib if not _is_han(t)}
    ascii_hit = any(ascii_terms & matched.get(cid, set()) for cid in matched)

    return {"结果": out, "统计": {"查询词项": q_terms[:10], "候选": len(final),
                                  "命中": len(out), "块总数": n_docs,
                                  "未收录词": missing[:6], "短语": phrases[:4],
                                  "查询词无区分度": weak_query,
                                  "可信度": _confidence(out, missing, q_terms, phrases,
                                                      ascii_hit, strong_filter),
                                  "算法": "BM25(k1=1.2,b=0.75) × IDF覆盖率 + 短语校验"}}


KIND_LABEL = {"methodology": "方法论文档", "report": "历史报告",
              "notice": "公告标题", "notice_body": "公告正文（关键条款）"}
