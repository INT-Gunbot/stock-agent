"""盘中实时行情：东方财富 push2 接口直连。

为什么不用 akshare.stock_zh_a_spot_em：
它内部按 100 行翻 59 页**串行**拉取，实测 66 秒；同样的数据用 12 线程并发分页只要 1.1 秒。
实时面板 15 秒刷一次，66 秒的接口在这里等于不可用。

两类数据分开取，代价不同：
- 指定代码的报价走 ulist.np，一次请求 0.2 秒 —— 持仓表、榜单、个股问答用这个
- 全市场快照只为算出涨跌家数、涨跌停家数、总成交额，1.1 秒，60 秒缓存一次

口径说明（写进 UI 文案，别让用户误会）：
实时价只用于"现在值多少钱"；选股因子、均线、财务仍是收盘日线口径。
盘中拿实时价去重算因子，会让榜单在一天里反复跳变，那种"实时"没有决策价值。
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from src.rules import limit_pct as _limit_pct

CN = timezone(timedelta(hours=8))

# 沪深京 A 股全市场：沪主板 + 科创板 + 深主板 + 创业板 + 北交所
_FS_ALL = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
_FIELDS = "f2,f3,f4,f5,f6,f8,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21"
_HDR = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Referer": "https://quote.eastmoney.com/",
}

INDICES = [
    ("1.000001", "上证指数"),
    ("0.399001", "深证成指"),
    ("0.399006", "创业板指"),
    ("1.000300", "沪深300"),
    ("1.000905", "中证500"),
]

# 缓存
_spot = {"at": 0.0, "rows": [], "error": None}
_quote_cache: dict[str, tuple[float, dict]] = {}
_lock_at = 0


# ---------------------------------------------------------------- 基础
def _get(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers=_HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def ts_code_to_secid(ts_code: str) -> str:
    """600519.SH -> 1.600519, 300750.SZ -> 0.300750, 830799.BJ -> 0.830799"""
    code = ts_code.split(".")[0]
    suffix = ts_code.split(".")[-1].upper() if "." in ts_code else ""
    if not suffix:
        suffix = "SH" if code[0] in "69" else "BJ" if code[0] in "48" else "SZ"
    return ("1." if suffix == "SH" else "0.") + code


# 涨跌停幅度的唯一实现搬到了 src/rules.py。
# 这里曾有一份独立实现，写成"ST 一律 5%"，把创业板/科创板的 ST 股（注册制板块，
# 上限仍是 20%）记成了 5% —— 后果是盘中涨停家数虚高，行业视图里正常涨 20% 的票
# 被当成复权失真剔掉。规则只留一份，见 rules.py 顶部说明。
# _limit_pct 从顶部 import 进来，名字保持不变（tools.py 用 RT._limit_pct 调用它）。


def _f(v, default=None):
    """东财停牌时返回 '-'，统一成 None。"""
    if v in ("-", "", None):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f


def session_state(now: datetime | None = None) -> dict:
    """A 股交易时段。返回 (标识, 中文, 是否盘中)。"""
    now = now or datetime.now(CN)
    if now.weekday() >= 5:
        return {"key": "closed", "text": "周末休市", "live": False}
    hm = now.hour * 100 + now.minute
    if hm < 915:
        return {"key": "pre", "text": "盘前", "live": False}
    if hm < 930:
        return {"key": "auction", "text": "集合竞价", "live": True}
    if hm < 1130:
        return {"key": "live", "text": "交易中", "live": True}
    if hm < 1300:
        return {"key": "lunch", "text": "午间休市", "live": False}
    if hm < 1500:
        return {"key": "live", "text": "交易中", "live": True}
    return {"key": "closed", "text": "已收盘", "live": False}


# ---------------------------------------------------------------- 全市场快照
def _page(pn: int) -> list[dict]:
    q = urllib.parse.urlencode({
        "pn": pn, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f3", "fs": _FS_ALL, "fields": _FIELDS,
    }, safe=":,+")
    d = (_get(f"https://push2.eastmoney.com/api/qt/clist/get?{q}") or {}).get("data") or {}
    return d.get("diff") or []


def spot_all(max_age: float = 60.0) -> dict:
    """全市场快照（带缓存）。首次 1~2 秒，之后命中缓存即时返回。"""
    global _spot
    now = time.time()
    if _spot["rows"] and now - _spot["at"] < max_age:
        return {**_spot, "cached": True}
    try:
        total_pages = 60
        with ThreadPoolExecutor(max_workers=12) as ex:
            pages = list(ex.map(_page, range(1, total_pages + 1)))
        rows = [r for p in pages for r in p if r.get("f12")]
        if not rows:
            raise RuntimeError("快照为空")
        _spot = {"at": now, "rows": rows, "error": None}
    except Exception as e:
        # 网络抖动时退回旧数据，标出 stale，UI 会显示"上次成功时间"
        if not _spot["rows"]:
            _spot = {**_spot, "error": str(e)[:120]}
    return {**_spot, "cached": False}


def breadth(rows: list[dict]) -> dict:
    """涨跌家数 / 涨跌停 / 成交额。口径与收盘版 market.breadth 对齐。"""
    up = down = flat = lu = ld = touched_up = 0
    amt = 0.0
    for r in rows:
        pct = _f(r.get("f3"))
        if pct is None:
            continue
        amt += _f(r.get("f6"), 0.0) or 0.0
        if pct > 0.001:
            up += 1
        elif pct < -0.001:
            down += 1
        else:
            flat += 1
        code, name = str(r.get("f12")), str(r.get("f14") or "")
        lim = _limit_pct(code, name)
        if pct >= lim - 0.3:
            lu += 1
        if pct <= -(lim - 0.3):
            ld += 1
        hi = _f(r.get("f15"))
        price = _f(r.get("f2"))
        if hi and price and _f(r.get("f18")):
            # 触及涨停 = 最高价达到涨停价（昨收 × (1+lim)），与收盘版同口径
            if hi >= _f(r.get("f18")) * (1 + lim / 100) - 0.005:
                touched_up += 1
    total = up + down + flat
    return {
        "up": up, "down": down, "flat": flat, "total": total,
        "limit_up": lu, "limit_down": ld, "touched_up": touched_up,
        "seal_rate": round(lu / touched_up * 100, 1) if touched_up else None,
        "amount": amt,
        "up_ratio": round(up / total * 100, 1) if total else None,
    }


# ---------------------------------------------------------------- 指定代码报价
def quotes(ts_codes: list[str], max_age: float = 5.0) -> dict[str, dict]:
    """按 ts_code 批量取实时报价，一次请求。返回 {ts_code: {...}}。

    注意区分「没报价」和「停牌」：盘前 09:15 之前东财把所有价格清成 "-"，
    这时候 suspended 必须是 False，否则界面上会显示一堆停牌股。
    """
    want = [c for c in dict.fromkeys(ts_codes) if c]
    now = time.time()
    live_session = session_state()["live"]
    need = [c for c in want if c not in _quote_cache or now - _quote_cache[c][0] > max_age]
    if need:
        secids = ",".join(ts_code_to_secid(c) for c in need)
        q = urllib.parse.urlencode({"fltt": 2, "invt": 2, "secids": secids, "fields": _FIELDS},
                                   safe=":,+.")
        try:
            d = (_get(f"https://push2.eastmoney.com/api/qt/ulist.np/get?{q}") or {}).get("data") or {}
            for r in d.get("diff") or []:
                code, mk = str(r.get("f12")), r.get("f13")
                ts = f"{code}.{'SH' if mk == 1 else 'BJ' if code.startswith(('4', '8', '92')) else 'SZ'}"
                price = _f(r.get("f2"))
                pre = _f(r.get("f18"))
                _quote_cache[ts] = (now, {
                    "ts_code": ts,
                    "name": r.get("f14"),
                    "price": price,
                    "pct": _f(r.get("f3")),
                    "chg": _f(r.get("f4")),
                    "open": _f(r.get("f17")),
                    "high": _f(r.get("f15")),
                    "low": _f(r.get("f16")),
                    "pre_close": pre,
                    "vol": _f(r.get("f5")),
                    "vol_ratio": _f(r.get("f10")),
                    "turnover": _f(r.get("f8")),
                    "amount": _f(r.get("f6")),
                    "mv": _f(r.get("f20")),
                    "no_quote": price is None,
                    "suspended": bool(price is None and live_session),
                    "limit_pct": _limit_pct(code, str(r.get("f14") or "")),
                })
        except Exception:
            pass  # 单个股票取不到不影响整体，缺失的键下面会被过滤
    return {c: _quote_cache[c][1] for c in want if c in _quote_cache}


def indices() -> list[dict]:
    secids = ",".join(s for s, _ in INDICES)
    q = urllib.parse.urlencode({"fltt": 2, "invt": 2, "secids": secids,
                                "fields": "f2,f3,f4,f6,f12,f14"}, safe=":,+.")
    try:
        d = (_get(f"https://push2.eastmoney.com/api/qt/ulist.np/get?{q}") or {}).get("data") or {}
        out = []
        for r in d.get("diff") or []:
            amt = _f(r.get("f6"))
            # 盘前东财把成交额和涨跌幅清零，此时 f2 是上一交易日收盘点位。
            # 直接显示 "+0.00%" 会让人以为今天平盘，所以单独打 pre 标记。
            pre = not amt
            out.append({
                "code": str(r.get("f12")), "name": r.get("f14"),
                "close": _f(r.get("f2")),
                "pct": None if pre else _f(r.get("f3")),
                "chg": None if pre else _f(r.get("f4")),
                "amount": amt,
                "pre": pre,
            })
        return out
    except Exception:
        return []


def snapshot(ts_codes: list[str] | None = None, with_breadth: bool = True) -> dict:
    """给 UI / Agent 的统一快照。"""
    ss = session_state()
    idx = indices()
    q = quotes(ts_codes or [])
    out = {
        "ts": datetime.now(CN).strftime("%Y-%m-%d %H:%M:%S"),
        "session": ss,
        "indices": idx,
        "quotes": q,
        "breadth": None,
        "pre_open": any(i.get("pre") for i in idx),
        "stale": False,
        "note": None,
    }
    if with_breadth:
        sp = spot_all()
        if sp.get("rows"):
            b = breadth(sp["rows"])
            # 09:15 之前东财会把报价清成 "-"（盘前重置），此时 total=0。
            # 直接给 0 涨 0 跌是错的，宁可返回空让上层回落到上一交易日收盘口径。
            if b.get("total"):
                out["breadth"] = b
                out["stale"] = time.time() - sp["at"] > 180
                out["spot_at"] = datetime.fromtimestamp(sp["at"], CN).strftime("%H:%M:%S")
                # 交易时段但全市场成交额为 0 —— 大概率是节假日
                if ss["live"] and (b.get("amount") or 0) < 1e8 and ss["key"] != "auction":
                    out["session"] = {"key": "closed", "text": "休市（非交易日）", "live": False}
            else:
                out["note"] = "盘前接口尚未生成报价，涨跌家数改用上一交易日收盘口径"
        elif sp.get("error"):
            out["note"] = f"全市场快照获取失败：{sp['error']}"
    return out


def market_label(snap: dict) -> str:
    """给 LLM 用的一句话数据口径说明。"""
    ss = snap.get("session") or {}
    if ss.get("live"):
        return f"盘中实时（{snap.get('ts','')}），行情为当前时点快照，约 15 秒延迟以内"
    return f"{ss.get('text','休市')}，行情为最近一次成交的快照（{snap.get('ts','')}）"
