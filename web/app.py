"""炒股 Agent 本地 demo 服务（仅标准库，无额外依赖）。

启动：
    python -m web.app            # 默认 127.0.0.1:8765

接口：
    GET  /api/all          一次性返回市场、选股、体检、持仓、日报要点
    GET  /api/live         盘中实时行情（指数、涨跌家数、指定代码报价）
    POST /api/positions    保存 positions.csv
    POST /api/run          触发 daily / select / checkup（互斥，排队执行）
    GET  /api/status       任务运行状态与输出
    POST /api/chat         发起一轮对话，立即返回 tid
    GET  /api/chat?tid=    轮询对话结果（含工具调用轨迹）
    GET  /reports/<file>   直接取报告文件（日报 iframe 用）
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.config import DATA_DIR, REPORT_DIR  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
PORT = 8765

TASKS = {
    "daily": ["-m", "scripts.run_daily", "--update"],
    "daily_only": ["-m", "scripts.run_daily"],
    "select": ["-m", "scripts.run_select"],
    "checkup": ["-m", "scripts.run_checkup"],
    "factor": ["-m", "scripts.quality_check"],
}

_job = {"task": None, "status": "idle", "log": "", "started": None, "elapsed": None}
_job_lock = threading.Lock()
_cache: dict = {"at": 0.0, "market": None}


# ---------------------------------------------------------------- 数据组装
def _latest(pattern: str) -> Path | None:
    files = sorted(REPORT_DIR.glob(pattern))
    return files[-1] if files else None


def _daily_points(md: str) -> list[str]:
    lines = md.splitlines()
    try:
        i = lines.index("**今日要点**")
    except ValueError:
        return []
    out = []
    for ln in lines[i + 1:]:
        s = ln.strip()
        if s.startswith("- "):
            out.append(s[2:].strip())
        elif not s:
            continue
        else:
            break
    return out


def _market() -> dict | None:
    """市场快照，60 秒缓存（广度查询要扫当日全表）。"""
    now = time.time()
    if _cache["market"] and now - _cache["at"] < 60:
        return _cache["market"]
    try:
        from src.db import connect
        from src.market import build as market_build
        from src.query import latest_trade_date

        con = connect(read_only=True)
        try:
            td = latest_trade_date(con)
            mk = market_build(con, td)
        finally:
            con.close()
        _cache.update(at=now, market=mk)
        return mk
    except Exception as e:  # 数据库被写入进程占用时会走到这里
        return {"error": str(e)[:160]}


def _select() -> dict | None:
    path = DATA_DIR / "recommendations.jsonl"
    if not path.exists():
        return None
    last = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def _checkup() -> dict | None:
    files = sorted(DATA_DIR.glob("checkup_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _positions() -> list[dict]:
    path = BASE / "positions.csv"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    return [dict(r) for r in csv.DictReader(io.StringIO(text))]


def _save_positions(rows: list[dict]) -> int:
    cols = ["ts_code", "name", "shares", "cost", "open_date", "tag", "note"]
    # 必须显式指定 lineterminator="\n"：DictWriter 默认写 "\r\n"，
    # 而 Path.write_text 在 Windows 上又会把 "\n" 转成 "\r\n"，
    # 两者叠加变成 "\r\r\n"，CSV 里会插入空行（Excel 打开可见）。
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    clean = []
    for r in rows:
        code = str(r.get("ts_code", "")).strip()
        if not code:
            continue
        clean.append({k: str(r.get(k, "") or "").strip() for k in cols})
        w.writerow(clean[-1])
    (BASE / "positions.csv").write_text(buf.getvalue(), encoding="utf-8-sig")
    return len(clean)


def _daily() -> dict:
    md_path = _latest("daily_*.md")
    html_path = _latest("daily_*.html")
    out = {"md_name": None, "html_name": None, "points": [], "date": None}
    if md_path:
        md = md_path.read_text(encoding="utf-8")
        out["md_name"] = md_path.name
        out["date"] = re.sub(r"\D", "", md_path.stem)[-8:]
        out["points"] = _daily_points(md)
    if html_path:
        out["html_name"] = html_path.name
    return out


def api_all() -> dict:
    cfg = _chat_cfg()
    rag_stats = None
    try:
        from src.agent import rag as R
        rag_stats = R.stats()
    except Exception:
        pass
    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        "market": _market(),
        "daily": _daily(),
        "select": _select(),
        "checkup": _checkup(),
        "positions": _positions(),
        "job": _job,
        "tasks": sorted(TASKS),
        "chat": {"enabled": cfg["enabled"], "model": cfg["model"]},
        "rag": rag_stats,
    }


# ---------------------------------------------------------------- 实时行情
def api_live(codes: list[str] | None = None) -> dict:
    """盘中实时快照。

    默认带上持仓 + 榜单前 5 的代码，前端一次请求就能把表格里的现价刷掉。
    """
    from src import realtime as RT

    if not codes:
        codes = [r.get("ts_code", "").strip() for r in _positions() if r.get("ts_code")]
        picks = _select() or {}
        codes += [p["ts_code"] for p in (picks.get("picks") or [])[:5]]
    try:
        return RT.snapshot([c for c in codes if c], with_breadth=True)
    except Exception as e:
        return {"error": str(e)[:160], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---------------------------------------------------------------- 对话
def _chat_cfg() -> dict:
    from src.agent import chat as CHAT
    return CHAT.cfg()


def chat_stream_to(handler, body: dict) -> None:
    """流式对话：一个 POST 请求，响应体是 NDJSON 事件流。

    为什么用 NDJSON 而不是 SSE：SSE 要走 EventSource，它只能发 GET 不能带请求体，
    于是问题得塞进 URL 或者先 POST 再 GET 两步。而这里本来就是"发一个问题、收一串事件"，
    用 fetch + 逐行读 body 最直接，也不会有 EventSource 自动重连带来的重复请求。
    """
    from src.agent import chat as CHAT
    from src.agent import store as ST

    q = str(body.get("question") or "").strip()
    if not q:
        handler._json({"error": "问题为空"}, 400)
        return

    # 联网开关由前端传。缺省为开：老页面或脚本调用不带这个字段时行为不变。
    # 只认 False 一种关闭值——不做 "false"/"0"/"off" 的宽松解析，
    # 否则哪天前后端布尔口径不一致，功能会被静默关掉，排查起来很难。
    allow_web = body.get("allow_web", True) is not False

    sid = str(body.get("session_id") or "").strip()
    if sid and not ST.get_session(sid):
        sid = ""
    fresh = not sid
    if fresh:
        sid = ST.create()["id"]

    ST.add(sid, "user", q)
    ST.auto_title(sid, q)
    history = ST.history(sid, limit=12)

    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()

    def emit(ev: dict) -> None:
        ev = dict(ev)
        if ev.get("t") == "meta":
            ev["session_id"] = sid
            ev["fresh"] = fresh
        try:
            handler.wfile.write((json.dumps(ev, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
            handler.wfile.flush()
        except Exception:
            pass  # 客户端断开（点了停止/关页面），不要因此让整个任务崩掉

    try:
        r = CHAT.chat_stream(history, emit=emit, allow_web=allow_web)
    except Exception as e:
        r = {"answer": f"出错了：{type(e).__name__}: {e}", "mode": "error"}
        emit({"t": "done", "answer": r["answer"], "mode": "error", "session_id": sid})

    # 落库放在流之后：中途断开时答案可能不完整，但至少不会写进半条记录
    try:
        ST.add(sid, "assistant", r.get("answer") or "", {
            "model": r.get("model"), "mode": r.get("mode"), "elapsed": r.get("elapsed"),
            "trace": r.get("trace"), "citations": r.get("citations"),
            "plan": r.get("plan"), "plan_type": r.get("plan_type"),
            "usage": r.get("usage"), "web": r.get("web"),
        })
    except Exception:
        pass


def api_sessions() -> dict:
    from src.agent import store as ST
    return {"sessions": ST.list_sessions()}


def api_session(sid: str) -> dict:
    from src.agent import store as ST
    s = ST.get_session(sid)
    if not s:
        return {"error": "会话不存在"}
    return {"session": s, "messages": ST.messages(sid)}



# ---------------------------------------------------------------- K 线
def api_kline(qs: dict) -> dict:
    """K 线数据。参数全部有默认值，前端一次拿全量再由 canvas 切片。

    为什么不在这里分页：3 年 726 根 × 12 列也只有 100KB 左右，
    一次取完前端可以随意缩放拖动而不用每拖一下就请求一次。
    """
    from src import kline as K

    code = (qs.get("code") or ["600519"])[0].strip()
    try:
        n = int((qs.get("n") or ["750"])[0])
    except ValueError:
        n = 750
    period = (qs.get("period") or ["day"])[0]
    adjust = (qs.get("adjust") or ["qfq"])[0]
    try:
        r = K.build(code, n=n, period=period, adjust=adjust,
                    with_live=(qs.get("live") or ["1"])[0] != "0")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200], "query": code}
    # 候选列表顺带给个默认名，前端下拉能直接显示
    return r


def api_etf_screen(qs: dict) -> dict:
    """ETF 筛选。与选股页刻意分开：个股那套评分吃 ROE / 净利 / PE 分位，
    ETF 这三样都没有，混进去只会退化成"涨幅榜"。这里按 ETF 自己的四项
    （规模、流动性、折溢价、同类相对强弱）排。
    """
    from src import etf as E

    def _num(key, default=None, cast=float):
        v = (qs.get(key) or [""])[0].strip()
        if v == "":
            return default
        try:
            return cast(v)
        except ValueError:
            return default

    category = (qs.get("category") or [""])[0].strip() or None
    kw = (qs.get("kw") or [""])[0].strip() or None
    sort = (qs.get("sort") or ["amt"])[0].strip()
    # 传 "none" 表示不排除任何分类（含货币/债券）。不能用空串——
    # http.server 的 parse_qs 默认 keep_blank_values=False，`exclude=` 会被
    # 整条丢掉，屏幕上看就是"选了含货基但货基还是不出现"。
    ex_raw = (qs.get("exclude") or ["货币,债券"])[0].strip()
    exclude = None if ex_raw.lower() in ("none", "all", "-") else (
        tuple(x for x in ex_raw.split(",") if x.strip()) or None)
    try:
        return E.screen(
            category=category,
            kw=kw,
            min_amount=_num("min_amt", 500.0),
            min_size=_num("min_size", 0.0),
            max_premium=_num("max_prem", None),
            sort=sort,
            limit=_num("limit", 80, int) or 80,
            exclude=exclude,
            refresh=(qs.get("refresh") or ["1"])[0] != "0",
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


# ---------------------------------------------------------------- 任务执行
def _run_job(task: str) -> None:
    cmd = [sys.executable, "-u"] + TASKS[task]
    with _job_lock:
        _job.update(task=task, status="running", log="", started=time.time(), elapsed=None)
    try:
        p = subprocess.run(cmd, cwd=str(BASE), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=3600)
        log = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        log = "\n".join(ln for ln in log.splitlines() if "it/s]" not in ln)
        status = "done" if p.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        log, status = "任务超时（>60 分钟）", "failed"
    except Exception as e:
        log, status = f"启动失败：{e}", "failed"
    with _job_lock:
        _job.update(status=status, log=log[-4000:], elapsed=round(time.time() - _job["started"], 1))
    _cache.update(at=0.0, market=None)   # 数据变了，缓存失效


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "StockAgentDemo/1.0"

    def log_message(self, *args):  # 静默访问日志，只保留自己的 print
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/all":
            self._json(api_all())
            return
        if path == "/api/live":
            codes = [c for c in (qs.get("codes") or [""])[0].split(",") if c.strip()]
            self._json(api_live(codes or None))
            return
        if path == "/api/kline":
            self._json(api_kline(qs))
            return
        if path == "/api/etf":
            self._json(api_etf_screen(qs))
            return
        if path == "/api/chat":
            cfg = _chat_cfg()
            self._json({"enabled": cfg["enabled"], "model": cfg["model"]})
            return
        if path == "/api/sessions":
            sid = (qs.get("id") or [""])[0].strip()
            self._json(api_session(sid) if sid else api_sessions())
            return
        if path == "/api/rag":
            from src.agent import rag as R
            try:
                self._json(R.stats())
            except Exception as e:
                self._json({"error": str(e)[:160]})
            return
        if path == "/api/status":
            with _job_lock:
                self._json(dict(_job))
            return
        if path.startswith("/reports/"):
            name = path.split("/")[-1]
            target = REPORT_DIR / name
            if target.exists() and target.parent == REPORT_DIR:
                ctype = "text/html; charset=utf-8" if name.endswith(".html") else "text/plain; charset=utf-8"
                self._send(200, target.read_bytes(), ctype)
                return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid json"}, 400)
            return

        if path == "/api/positions":
            try:
                n = _save_positions(body.get("rows") or [])
                self._json({"ok": True, "saved": n})
            except Exception as e:
                self._json({"error": str(e)[:200]}, 500)
            return

        if path == "/api/chat":
            chat_stream_to(self, body)
            return

        if path == "/api/sessions":
            from src.agent import store as ST
            act = body.get("action") or "create"
            try:
                if act == "create":
                    self._json({"ok": True, "session": ST.create(body.get("title") or "新对话")})
                elif act == "rename":
                    ok = ST.rename(str(body.get("id") or ""), str(body.get("title") or ""))
                    self._json({"ok": ok})
                elif act == "delete":
                    ok = ST.delete(str(body.get("id") or ""))
                    self._json({"ok": ok})
                else:
                    self._json({"error": f"未知操作：{act}"}, 400)
            except Exception as e:
                self._json({"error": str(e)[:200]}, 500)
            return

        if path == "/api/run":
            task = body.get("task")
            if task not in TASKS:
                self._json({"error": f"未知任务：{task}"}, 400)
                return
            with _job_lock:
                if _job["status"] == "running":
                    self._json({"error": f"已有任务在跑：{_job['task']}"}, 409)
                    return
                _job.update(task=task, status="running", log="", started=time.time())
            threading.Thread(target=_run_job, args=(task,), daemon=True).start()
            self._json({"ok": True, "task": task})
            return

        self._json({"error": "not found"}, 404)


URL = f"http://127.0.0.1:{PORT}"


def _port_busy(port: int = PORT) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    no_browser = "--no-browser" in sys.argv

    # 已经在跑：不重复起服务，直接把页面拉出来
    if _port_busy():
        print(f"[demo] 已在运行：{URL}")
        if not no_browser:
            import webbrowser

            webbrowser.open(URL)
        return

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[demo] {URL}   （Ctrl+C 停止）")
    if not no_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(URL)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] 已停止")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
