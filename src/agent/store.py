"""会话持久化：SQLite 独立文件，一行一条消息。

为什么不用 DuckDB：
    主库是单文件写入独占的（更新行情时锁库），而会话是**高频小写**——
    每问一句要写两条记录。放进主库会让"正在更新数据"和"正在对话"互相阻塞。
    所以单独一个 data/chat.db，WAL 模式，读写不互斥。

为什么不用 JSON 文件整存整取：
    每次追加消息都要重写整个文件，会话一多月就变成 O(n) 写放大，
    且并发两个标签页时会丢消息。SQLite 的 INSERT 天然是原子的。

表结构刻意扁平：消息表里 meta 字段存 JSON（trace / model / usage / citations），
因为这些都是"回看时才需要"的展示信息，不需要按字段查询。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from src.config import DATA_DIR

DB = DATA_DIR / "chat.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '新对话',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    model       TEXT
);
CREATE TABLE IF NOT EXISTS message (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    meta        TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES session(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON message(session_id, id);
CREATE INDEX IF NOT EXISTS idx_sess_updated ON session(updated_at DESC);
"""


def _con() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB), timeout=5.0)
    con.row_factory = sqlite3.Row
    # WAL：读写并发不互斥。第二个参数做 NORMAL 同步，够用且快很多。
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(_SCHEMA)
    return con


# ---------------------------------------------------------------- 会话
def create(title: str = "新对话", model: str | None = None) -> dict:
    sid = uuid.uuid4().hex[:12]
    now = time.time()
    con = _con()
    try:
        con.execute("INSERT INTO session (id, title, created_at, updated_at, model) VALUES (?,?,?,?,?)",
                    [sid, title[:40], now, now, model])
        con.commit()
    finally:
        con.close()
    return {"id": sid, "title": title[:40], "created_at": now, "updated_at": now, "model": model}


def list_sessions(limit: int = 50) -> list[dict]:
    con = _con()
    try:
        rows = con.execute(
            """SELECT s.id, s.title, s.created_at, s.updated_at, s.model,
                      (SELECT count(*) FROM message m WHERE m.session_id = s.id) AS n
               FROM session s ORDER BY s.updated_at DESC LIMIT ?""", [limit]
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def get_session(sid: str) -> dict | None:
    con = _con()
    try:
        r = con.execute("SELECT * FROM session WHERE id = ?", [sid]).fetchone()
    finally:
        con.close()
    return dict(r) if r else None


def rename(sid: str, title: str) -> bool:
    con = _con()
    try:
        con.execute("UPDATE session SET title = ? WHERE id = ?", [title.strip()[:40] or "新对话", sid])
        con.commit()
        return con.total_changes > 0
    finally:
        con.close()


def delete(sid: str) -> bool:
    con = _con()
    try:
        con.execute("DELETE FROM message WHERE session_id = ?", [sid])
        con.execute("DELETE FROM session WHERE id = ?", [sid])
        con.commit()
        return con.total_changes > 0
    finally:
        con.close()


# ---------------------------------------------------------------- 消息
def add(sid: str, role: str, content: str, meta: dict | None = None) -> int:
    now = time.time()
    con = _con()
    try:
        cur = con.execute(
            "INSERT INTO message (session_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
            [sid, role, content, json.dumps(meta, ensure_ascii=False, default=str) if meta else None, now],
        )
        con.execute("UPDATE session SET updated_at = ? WHERE id = ?", [now, sid])
        con.commit()
        mid = int(cur.lastrowid or 0)
    finally:
        con.close()
    return mid


def messages(sid: str, limit: int = 200) -> list[dict]:
    con = _con()
    try:
        rows = con.execute(
            "SELECT id, role, content, meta, created_at FROM message WHERE session_id = ? "
            "ORDER BY id ASC LIMIT ?", [sid, limit]
        ).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("meta"):
            try:
                d["meta"] = json.loads(d["meta"])
            except Exception:
                d["meta"] = None
        out.append(d)
    return out


def history(sid: str, limit: int = 12) -> list[dict]:
    """给模型用的上下文：只保留 user/assistant 的纯文本，最近 N 条。"""
    con = _con()
    try:
        rows = con.execute(
            "SELECT role, content FROM message WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            [sid, limit],
        ).fetchall()
    finally:
        con.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)
            if r["role"] in ("user", "assistant") and (r["content"] or "").strip()]


def auto_title(sid: str, first_q: str) -> str | None:
    """用第一句话当前 18 字标题。不做 LLM 摘要——那要额外一次调用，不值。"""
    s = get_session(sid)
    if not s or s["title"] not in ("新对话", ""):
        return None
    t = " ".join(str(first_q).split())[:18]
    if not t:
        return None
    rename(sid, t)
    return t
