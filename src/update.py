"""数据更新编排：增量拉取 + 入库 + 记录日志。

两种增量方式，代价差两个数量级：

- `update_daily`：逐票拉日线（每票 2 次请求：东财日线 + 新浪因子），5678 只 ≈ 9-20 分钟。
  只有**补历史缺口**时才需要它 —— 快照只能给你"今天"，给不了上周三。
- `update_daily_snapshot`：一次快照拿全市场当日 OHLCV，1-2 秒。
  当日 K 线本来就是截面数据，按票去拉是拿点查代替扫描。

`update_daily_auto` 负责在两者之间选，调用方不用关心。
"""
from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import query as Q
from . import realtime as RT
from .config import DEFAULT_WORKERS, FULL_HISTORY_YEARS, INCREMENT_LOOKBACK_DAYS
from .db import connect, exec_dml, init_schema, scalar, upsert_anti_join
from .sources import akshare_source as src

INDEX_CODES = {
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
    "000300": "沪深300",
}


def _log(con, task, target, status, rows, message, started, finished):
    exec_dml(
        con,
        "INSERT INTO update_log VALUES (?,?,?,?,?,?,?)",
        [task, target, status, rows, message, started, finished],
    )


def _range(full: bool) -> tuple[str, str]:
    today = dt.date.today()
    if full:
        start = today - dt.timedelta(days=365 * FULL_HISTORY_YEARS)
    else:
        start = today - dt.timedelta(days=INCREMENT_LOOKBACK_DAYS)
    return start.strftime("%Y%m%d"), today.strftime("%Y%m%d")


def update_stock_list(verbose: bool = True) -> int:
    con = connect()
    init_schema(con)
    started = dt.datetime.now()
    try:
        df = src.fetch_stock_list()
        con.register("_staging_dim", df)
        exec_dml(
            con,
            "INSERT INTO dim_stock SELECT t.* FROM _staging_dim t "
            "LEFT JOIN dim_stock d ON d.ts_code = t.ts_code WHERE d.ts_code IS NULL",
        )
        con.unregister("_staging_dim")
        n = scalar(con, "SELECT count(*) FROM dim_stock")
        _log(con, "dim_stock", "all", "ok", len(df), f"总数量 {n}", started, dt.datetime.now())
        if verbose:
            print(f"[股票列表] 上送 {len(df)} 条，库内合计 {n} 条")
        return len(df)
    except Exception as e:
        _log(con, "dim_stock", "all", "fail", 0, str(e)[:300], started, dt.datetime.now())
        raise
    finally:
        con.close()


def _fetch_one(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一次返回 (不复权日线, 复权因子)。因子由后复权价与原价反推。"""
    return src.fetch_daily(symbol, start, end)


def update_daily(codes: list[str] | None = None,
                 full: bool = False,
                 workers: int = DEFAULT_WORKERS,
                 verbose: bool = True) -> dict:
    """更新日线与复权因子。

    codes: 6 位代码或 ts_code 列表；None 表示全市场。
    """
    con = connect()
    init_schema(con)
    started = dt.datetime.now()
    start, end = _range(full)

    if codes is None:
        syms = con.execute("SELECT symbol FROM dim_stock WHERE is_active ORDER BY symbol").df()["symbol"].tolist()
    else:
        syms = [str(c).split(".")[0].zfill(6) for c in codes]

    ok, fail, new_bars, new_facs = 0, [], 0, 0
    done = 0
    total = len(syms)
    if verbose:
        print(f"[日线] 待更新 {total} 只，区间 {start}~{end}，并发 {workers}")

    def _work(sym):
        return sym, _fetch_one(sym, start, end)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_work, s): s for s in syms}
        for fut in as_completed(futures):
            sym = futures[fut]
            done += 1
            try:
                _, (bar, fac) = fut.result()
                if not bar.empty:
                    new_bars += upsert_anti_join(con, bar, "daily_bar", ["ts_code", "trade_date"])
                if not fac.empty:
                    new_facs += upsert_anti_join(con, fac, "adj_factor", ["ts_code", "trade_date"])
                ok += 1
            except Exception as e:
                fail.append((sym, str(e)[:120]))
            if verbose and done % 50 == 0:
                print(f"  进度 {done}/{total}  新增K线 {new_bars}  失败 {len(fail)}")

    _log(con, "daily", f"{start}~{end}", "ok" if not fail else "partial",
         new_bars, f"ok={ok} fail={len(fail)}", started, dt.datetime.now())
    con.close()

    if verbose:
        print(f"[日线] 完成：成功 {ok} 只，失败 {len(fail)} 只，新增日线 {new_bars} 行，新增复权因子 {new_facs} 行")
        for s, msg in fail[:5]:
            print(f"  失败样例 {s}: {msg}")
    return {"ok": ok, "fail": fail, "new_bars": new_bars, "new_factors": new_facs}


# ---------------------------------------------------------------- 快照增量（秒级）
_TD_CACHE: dict = {"at": 0.0, "days": set()}


def trade_days(refresh: bool = False) -> set[dt.date]:
    """A 股交易日历（新浪，覆盖到当年年末）。缓存 12 小时。

    为什么需要它：周末/节假日跑更新时，东财快照返回的是**上一个交易日**的收盘价，
    如果直接按"今天"入库，周五的数据会被写成周六的 K 线，全库日期错位。
    """
    now = time.time()
    if not refresh and _TD_CACHE["days"] and now - _TD_CACHE["at"] < 43200:
        return _TD_CACHE["days"]
    try:
        import akshare as ak

        df = ak.tool_trade_date_hist_sina()
        days = {pd.to_datetime(d).date() for d in df["trade_date"]}
        _TD_CACHE.update(at=now, days=days)
        return days
    except Exception:
        return _TD_CACHE["days"] or set()


def _num(v, default=None):
    if v in ("-", "", None):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def bars_from_snapshot(rows: list[dict], trade_date: dt.date) -> pd.DataFrame:
    """东财全市场快照 -> daily_bar 行（列与单位对齐既有数据）。

    单位核对过：快照 f5 成交量是「手」、f6 成交额是「元」，与 akshare 日线一致
    （茅台 82691 手 × 1439 元 ≈ 118.5 亿元，和库里存的对得上）。
    """
    recs = []
    for r in rows:
        code = str(r.get("f12") or "")
        if not code:
            continue
        close = _num(r.get("f2"))
        pre = _num(r.get("f18"))
        amt = _num(r.get("f6"))
        # 当天没成交（停牌 / 未开盘）就不该有 K 线，与逐票口径保持一致
        if not close or not pre or not amt:
            continue
        high, low, open_ = _num(r.get("f15")), _num(r.get("f16")), _num(r.get("f17"))
        if not high or not low or not open_:
            continue
        vol = _num(r.get("f5"), 0.0)
        ts = f"{code}.{'SH' if r.get('f13') == 1 else 'BJ' if code.startswith(('4', '8', '92')) else 'SZ'}"
        recs.append({
            "ts_code": ts,
            "trade_date": trade_date,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pre,
            # 振幅用快照自算，避免再要一个字段：东财口径就是 (高-低)/昨收
            "amplitude": round((high - low) / pre * 100, 2) if pre else None,
            "vol": vol,
            "amount": amt,
            "pct_chg": _num(r.get("f3")),
            "price_chg": _num(r.get("f4")),
            "turnover": _num(r.get("f8")),
        })
    if not recs:
        return pd.DataFrame(columns=src.BAR_COLS)
    df = pd.DataFrame(recs)
    for c in ("open", "high", "low", "close", "pre_close", "vol", "amount",
              "amplitude", "pct_chg", "price_chg", "turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["ts_code", "trade_date"] + [c for c in src.BAR_COLS if c not in ("ts_code", "trade_date")]]


def _write_bars(con, df: pd.DataFrame) -> int:
    """按主键 upsert 日线。

    不能用 upsert_anti_join：盘中跑过一次写入的是半根 K 线，
    收盘后再跑如果只做 anti-join，那根错的数据永远修不掉。
    """
    if df is None or df.empty:
        return 0
    con.register("_staging_bar", df)
    try:
        exec_dml(
            con,
            """
            INSERT INTO daily_bar BY NAME SELECT * FROM _staging_bar
            ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                open = excluded.open, high = excluded.high, low = excluded.low,
                close = excluded.close, pre_close = excluded.pre_close,
                vol = excluded.vol, amount = excluded.amount,
                amplitude = excluded.amplitude, pct_chg = excluded.pct_chg,
                price_chg = excluded.price_chg, turnover = excluded.turnover
            """,
        )
    finally:
        con.unregister("_staging_bar")
    return len(df)


def _inherit_adj_factor(con, trade_date: dt.date) -> int:
    """把每只票「最近一次已知因子」继承到当日。

    复权因子是阶梯函数：没除权就不跳变。逐票重拉一遍只为确认"今天没除权"，
    纯属浪费 —— 改成先继承，再用涨跌幅越界把可疑票挑出来单独校准。
    """
    exec_dml(
        con,
        """
        INSERT INTO adj_factor
        SELECT b.ts_code, b.trade_date, f.adj_factor
        FROM daily_bar b
        JOIN (
            SELECT DISTINCT ON (ts_code) ts_code, adj_factor
            FROM adj_factor
            WHERE trade_date < ?
            ORDER BY ts_code, trade_date DESC
        ) f ON f.ts_code = b.ts_code
        WHERE b.trade_date = ?
        ON CONFLICT (ts_code, trade_date) DO UPDATE SET adj_factor = excluded.adj_factor
        """,
        [trade_date, trade_date],
    )
    # 兜底：当天有 K 线、但历史上从没有过因子的，只能是新股上市首日（因子恒为 1）。
    # 不补这一步，新股的 adj_close 会是 NULL，均线和动量直接算不出来。
    exec_dml(
        con,
        """
        INSERT INTO adj_factor
        SELECT b.ts_code, b.trade_date, 1.0
        FROM daily_bar b
        LEFT JOIN adj_factor a ON a.ts_code = b.ts_code AND a.trade_date = b.trade_date
        WHERE b.trade_date = ? AND a.ts_code IS NULL
        ON CONFLICT (ts_code, trade_date) DO NOTHING
        """,
        [trade_date],
    )
    return int(scalar(con, "SELECT count(*) FROM adj_factor WHERE trade_date = ?", [trade_date]) or 0)


def find_exdiv_candidates(con, trade_date: dt.date) -> list[tuple[str, str, float]]:
    """找出当日大概率除权的票：跌幅越过了板块跌停上限。

    判据为什么不用「昨收与前收不符」：实测东财的 pre_close **永远**等于前一交易日
    收盘（茅台 2024-06-19 除权日当天，pre_close 与 06-18 收盘的差值也是 0），
    它的 pct_chg 就是含除权缺口的原始跌幅。所以拿昨收比对检测不出任何除权。

    于是退回硬约束：一天能跌多少是交易所定死的。跌得比跌停还深，只有一种解释 ——
    那是除权缺口，不是真实下跌。缓冲 0.6 个百分点吸收了价格四舍五入。

    不用「名称含 ST 就按 5%」：dim_stock.name 是**当前**名称，拿它去判历史涨跌停
    必然出错（实测 *ST亚士 在戴帽前跌了 -9.9%，会误判成除权）。
    主板统一按 -10.6% 卡，代价是漏掉主板 ST 股的小额送转 —— 而 ST 股基本不送转。
    """
    rows = con.execute(
        """
        SELECT b.ts_code, d.name, b.pct_chg
        FROM daily_bar b LEFT JOIN dim_stock d ON d.ts_code = b.ts_code
        WHERE b.trade_date = ?
          AND b.pct_chg < -(CASE
                WHEN b.ts_code LIKE '%.BJ' THEN 30.0
                WHEN left(b.ts_code, 3) IN ('300', '301', '302', '688', '689') THEN 20.0
                ELSE 10.0 END) - 0.6
        ORDER BY b.pct_chg
        """,
        [trade_date],
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _fix_exdiv(con, trade_date: dt.date, workers: int = 4, verbose: bool = True) -> dict:
    """只对疑似除权的票重拉因子（通常几十只，秒级）。

    用 upsert 而不是 UPDATE：新股上市首日在 adj_factor 里本来就没有行，
    UPDATE 会静默失效，因子永远补不上。
    """
    cands = find_exdiv_candidates(con, trade_date)
    if not cands:
        return {"checked": 0, "fixed": 0, "names": []}

    fixed, names = 0, []

    def _one(ts_code):
        return ts_code, src.fetch_adj_factor(ts_code.split(".")[0], [trade_date])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(_one, c[0]) for c in cands]):
            try:
                ts_code, fac = fut.result()
                if fac is None or fac.empty:
                    continue
                con.register("_staging_fac", fac)
                try:
                    exec_dml(
                        con,
                        """
                        INSERT INTO adj_factor BY NAME SELECT * FROM _staging_fac
                        ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                            adj_factor = excluded.adj_factor
                        """,
                    )
                finally:
                    con.unregister("_staging_fac")
                fixed += 1
                names.append(ts_code)
            except Exception:
                continue
    if verbose and names:
        print(f"[除权] 疑似 {len(cands)} 只，因子已校准 {fixed} 只：{', '.join(names[:8])}"
              + (" …" if len(names) > 8 else ""))
    return {"checked": len(cands), "fixed": fixed, "names": names}


def _missing_days(con, trade_date: dt.date, cal: set[dt.date]) -> list[dt.date]:
    """库内最新日线之后、目标日之前，交易日历里还差哪几天。"""
    last = scalar(con, "SELECT max(trade_date) FROM daily_bar")
    if last is None:
        return [trade_date]
    return sorted(d for d in cal if last < d < trade_date)


def update_daily_snapshot(trade_date: dt.date | None = None,
                          force: bool = False,
                          fix_exdiv: bool = True,
                          verbose: bool = True) -> dict:
    """用一次全市场快照写入当日 K 线（1-2 秒）。

    force=True 允许盘中写入 —— 但要知道后果：写进去的是半根 K 线，
    价格、成交量、涨跌幅全都不是最终值。收盘后再跑一次会覆盖修正，
    可如果那天没再跑，库里就留着一根失真的 K 线。默认禁止。
    """
    now = dt.datetime.now()
    today = trade_date or now.date()
    cal = trade_days()

    if not force:
        if cal and today not in cal:
            return {"ok": False, "mode": "snapshot", "error": f"{today} 不是交易日，不写当日 K 线"}
        hm = now.hour * 100 + now.minute
        if today == now.date() and hm < 1500:
            ss = RT.session_state()
            return {"ok": False, "mode": "snapshot",
                    "error": f"当前{ss['text']}（{now.strftime('%H:%M')}），未收盘。"
                             f"盘中写入的是半根 K 线，收盘后再跑一次才会修正。"}

    t0 = time.time()
    sp = RT.spot_all(max_age=0)
    rows = sp.get("rows") or []
    if not rows:
        return {"ok": False, "mode": "snapshot", "error": f"快照为空：{sp.get('error') or '未知原因'}"}

    bar = bars_from_snapshot(rows, today)
    if bar.empty:
        return {"ok": False, "mode": "snapshot",
                "error": "快照里没有有效报价（盘前 09:15 前东财会清空价格）"}

    con = connect()
    init_schema(con)
    started = dt.datetime.now()
    try:
        active = int(scalar(con, "SELECT count(*) FROM dim_stock WHERE is_active") or 0)
        n_bar = _write_bars(con, bar)
        n_fac = _inherit_adj_factor(con, today)
        exdiv = _fix_exdiv(con, today, verbose=verbose) if fix_exdiv else {"checked": 0, "fixed": 0}

        # 只保留库内认识的票（快照可能包含退市整理、B 股等）
        n_new = int(scalar(con, "SELECT count(*) FROM daily_bar WHERE trade_date = ?", [today]) or 0)
        _log(con, "daily_snapshot", str(today), "ok", n_bar,
             f"bars={n_bar} fac={n_fac} exdiv={exdiv['fixed']}", started, dt.datetime.now())
        elapsed = round(time.time() - t0, 2)

        res = {
            "ok": True, "mode": "snapshot", "trade_date": str(today),
            "bars": n_new, "factors": n_fac, "active": active,
            "coverage": round(n_new / active * 100, 1) if active else None,
            "exdiv_checked": exdiv["checked"], "exdiv_fixed": exdiv["fixed"],
            "exdiv_names": exdiv.get("names", [])[:10],
            "seconds": elapsed, "new_bars": 0, "fail": [],
        }
        if verbose:
            print(f"[日线·快照] {today} 写入 {n_new} 只 / 活跃 {active} 只"
                  f"（覆盖 {res['coverage']}%），因子继承 {n_fac}，"
                  f"除权校准 {exdiv['fixed']}，耗时 {elapsed}s")
        return res
    finally:
        con.close()


def update_daily_auto(verbose: bool = True, fix_exdiv: bool = True) -> dict:
    """能走快照就走快照，缺历史才逐票。

    判断依据：库内最新日线到目标交易日之间，交易日历上还差几天。
    差 0 天 -> 快照；差 1 天以上 -> 逐票补齐（快照给不了"上周三"）。
    """
    today = dt.date.today()
    cal = trade_days()
    # 目标日 = 日历里 <= 今天的最后一个交易日。周末跑就落到周五，不会写错日期。
    target = max((d for d in cal if d <= today), default=today)

    con = connect(read_only=True)
    try:
        last = scalar(con, "SELECT max(trade_date) FROM daily_bar")
        gaps = _missing_days(con, target, cal)
    finally:
        con.close()

    if last is not None and last >= target:
        if verbose:
            print(f"[日线] 库内最新 {last} 已是最新交易日，无需更新")
        return {"ok": True, "mode": "skip", "trade_date": str(last), "bars": 0}

    if gaps:
        if verbose:
            print(f"[日线] 缺 {len(gaps)} 个交易日（{gaps[0]}~{gaps[-1]}），改走逐票补数")
        r = update_daily(full=False, workers=DEFAULT_WORKERS, verbose=verbose)
        r["mode"] = "per_stock"
        return r

    return update_daily_snapshot(trade_date=target, fix_exdiv=fix_exdiv, verbose=verbose)


def update_factors(codes: list[str] | None = None,
                   workers: int = 6,
                   delay: float = 0.0,
                   verbose: bool = True) -> dict:
    """重拉复权因子（新浪阶梯因子），只改 adj_factor 表，不动日线。

    为什么要单独有这个函数：
    早期用"东财后复权价 ÷ 原始价"反推因子，事后证明那条序列做过收益压缩，
    因子每日漂移、累计收益被系统性低估。换成新浪事件表后因子变成干净的阶梯，
    但存量数据是用旧方法灌的，必须整体重灌一次。

    做法：先在内存里拼出全量新因子表，再用 staging 表整体替换，
    避免逐只 DELETE/INSERT 产生 5678 次小事务，也避免中断后留下半新半旧的数据。
    """
    con = connect()
    init_schema(con)
    started = dt.datetime.now()

    if codes is None:
        rows = con.execute(
            "SELECT ts_code, symbol FROM dim_stock WHERE is_active ORDER BY symbol"
        ).fetchall()
    else:
        rows = [
            (src.to_ts_code(str(c).split(".")[0].zfill(6)), str(c).split(".")[0].zfill(6))
            for c in codes
        ]

    # 一次性取出每只票的交易日列表，避免 5678 次往返查询
    date_map = {
        r[0]: r[1]
        for r in con.execute(
            "SELECT ts_code, list(trade_date ORDER BY trade_date) FROM daily_bar GROUP BY 1"
        ).fetchall()
    }
    targets = [(tc, sym) for tc, sym in rows if tc in date_map]
    total = len(targets)
    if verbose:
        print(f"[复权因子] 待更新 {total} 只，数据源 新浪 hfq-factor，并发 {workers}")

    frames, fail, done = [], [], 0

    def _work(ts_code, sym):
        # 新浪对高并发不友好：并发 6 时约 7% 的票会返回空/超时，
        # 降并发 + 加延迟后全部可拉通。失败重跑时用 workers=2, delay=0.4。
        if delay:
            time.sleep(delay)
        return ts_code, src.fetch_adj_factor(sym, date_map[ts_code])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_work, tc, sym): tc for tc, sym in targets}
        for fut in as_completed(futures):
            tc = futures[fut]
            done += 1
            try:
                _, fac = fut.result()
                if fac is not None and not fac.empty:
                    frames.append(fac)
                else:
                    fail.append((tc, "无因子数据"))
            except Exception as e:
                fail.append((tc, str(e)[:120]))
            if verbose and done % 500 == 0:
                print(f"  进度 {done}/{total}  成功 {len(frames)}  失败 {len(fail)}")

    if not frames:
        con.close()
        raise RuntimeError("未取到任何因子数据，已放弃替换")

    new_fac = pd.concat(frames, ignore_index=True)
    new_fac["trade_date"] = pd.to_datetime(new_fac["trade_date"]).dt.date
    n_new = len(new_fac)

    # 落库：全市场时整表替换（最快）；指定代码时只替换这几只（避免误删其他票的因子）
    con.register("_staging_fac", new_fac)
    try:
        if codes is None:
            exec_dml(con, "DROP TABLE IF EXISTS adj_factor_new")
            # 必须显式建表。早先用 CREATE TABLE AS SELECT 重建，CTAS 不继承主键约束，
            # 结果整张表变成无约束裸表：ON CONFLICT 直接失效，重复行会静默写进来，
            # 后面所有 join adj_factor 的地方都会被放大成笛卡尔积。
            exec_dml(con, """
                CREATE TABLE adj_factor_new (
                    ts_code     VARCHAR NOT NULL,
                    trade_date  DATE NOT NULL,
                    adj_factor  DOUBLE,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """)
            # 同一 (票, 日) 若在 staging 里出现多次，只留一条，否则插不进去
            exec_dml(con, """
                INSERT INTO adj_factor_new
                SELECT ts_code, trade_date, any_value(adj_factor)
                FROM _staging_fac GROUP BY ts_code, trade_date
            """)
            exec_dml(con, "DROP TABLE IF EXISTS adj_factor_old")
            exec_dml(con, "ALTER TABLE adj_factor RENAME TO adj_factor_old")
            exec_dml(con, "ALTER TABLE adj_factor_new RENAME TO adj_factor")
            exec_dml(con, "DROP TABLE adj_factor_old")
        else:
            exec_dml(
                con,
                "DELETE FROM adj_factor WHERE ts_code IN "
                "(SELECT DISTINCT ts_code FROM _staging_fac)",
            )
            exec_dml(con, "INSERT INTO adj_factor SELECT * FROM _staging_fac")
    finally:
        con.unregister("_staging_fac")

    _log(con, "factor", "sina-hfq", "ok" if not fail else "partial",
         n_new, f"ok={len(frames)} fail={len(fail)}", started, dt.datetime.now())
    con.close()

    if verbose:
        print(f"[复权因子] 完成：{len(frames)} 只 / {n_new} 行，失败 {len(fail)} 只")
        for s, m in fail[:5]:
            print(f"  失败样例 {s}: {m}")
    return {"ok": len(frames), "fail": fail, "rows": n_new}


def update_index(full: bool = False, verbose: bool = True) -> int:
    con = connect()
    init_schema(con)
    started = dt.datetime.now()
    start, end = _range(full)
    added = 0
    for code in INDEX_CODES:
        try:
            df = src.fetch_index(code, start, end)
            if not df.empty:
                added += upsert_anti_join(con, df, "index_bar", ["index_code", "trade_date"])
            if verbose:
                print(f"[指数] {code} {INDEX_CODES[code]} 最新 {df['trade_date'].max() if not df.empty else 'NA'}")
        except Exception as e:
            if verbose:
                print(f"[指数] {code} 失败: {str(e)[:100]}")
    _log(con, "index", ",".join(INDEX_CODES), "ok", added, "", started, dt.datetime.now())
    con.close()
    return added


def recent_periods(n: int = 8, today: dt.date | None = None) -> list[str]:
    """最近 n 个已过披露节点的季度末，返回 YYYYMMDD 列表。"""
    today = today or dt.date.today()
    ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    res: list[dt.date] = []
    y = today.year
    while len(res) < n:
        for mm, dd in reversed(ends):
            d = dt.date(y, mm, dd)
            if d < today:
                res.append(d)
        y -= 1
    return [d.strftime("%Y%m%d") for d in res[:n]]


def update_fina(periods: list[str] | None = None, verbose: bool = True) -> int:
    """拉取业绩报表（含公告日），按报告期逐个入库。

    同一报告期重复拉取会生成新的 ingested_at 版本，天然保留追溯调整痕迹。
    """
    con = connect()
    init_schema(con)
    started = dt.datetime.now()
    periods = periods or recent_periods(8)
    # 业绩报表接口会返回新三板公司（同样以 43/83/87 开头，无法从代码区分），
    # 用 A 股权威列表过滤，避免把非上市公司混进因子池。
    valid = {r[0] for r in con.execute("SELECT ts_code FROM dim_stock").fetchall()}
    total = 0
    for p in periods:
        try:
            df = src.fetch_yjbb(p)
            if df.empty:
                if verbose:
                    print(f"[财务] {p} 无数据（可能尚未披露）")
                continue
            before = len(df)
            df = df[df["ts_code"].isin(valid)]
            n = upsert_anti_join(con, df, "fina_yjbb", ["ts_code", "end_date", "ingested_at"])
            total += n
            if verbose:
                print(f"[财务] {p} 原始 {before} 条，过滤后 {len(df)} 条，新增 {n} 条")
        except Exception as e:
            if verbose:
                print(f"[财务] {p} 失败: {str(e)[:120]}")
    _log(con, "fina", ",".join(periods), "ok", total, "", started, dt.datetime.now())
    con.close()
    return total


def update_market_activity(verbose: bool = True) -> int:
    """当日市场活跃度（涨跌家数 / 涨跌停家数）。"""
    con = connect()
    init_schema(con)
    started = dt.datetime.now()
    trade_date = Q.latest_trade_date(con) or dt.date.today()
    try:
        df = src.fetch_market_activity()
        rows = 0
        upsert_sql = """
            INSERT INTO market_snapshot (trade_date, metric, value, extra, computed_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT (trade_date, metric) DO UPDATE SET
                value = excluded.value,
                extra = excluded.extra,
                computed_at = excluded.computed_at
        """
        for _, r in df.iterrows():
            try:
                val = float(str(r["value"]).replace("%", "").replace(",", ""))
                exec_dml(con, upsert_sql, [trade_date, str(r["metric"]), val, None, dt.datetime.now()])
                rows += 1
            except Exception:
                exec_dml(con, upsert_sql, [trade_date, str(r["metric"]), None, str(r["value"]), dt.datetime.now()])
        _log(con, "market_activity", str(trade_date), "ok", rows, "", started, dt.datetime.now())
        if verbose:
            print(f"[市场] {trade_date} 活跃度 {rows} 项")
        return rows
    except Exception as e:
        if verbose:
            print(f"[市场] 失败: {str(e)[:120]}")
        _log(con, "market_activity", str(trade_date), "fail", 0, str(e)[:200], started, dt.datetime.now())
        return 0
    finally:
        con.close()
