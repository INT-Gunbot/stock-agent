"""ETF 数据层：与个股严格分表，但复权口径与 get_price 完全一致。

三条设计约定：

1. **独立表，不混进 dim_stock / daily_bar。** 日报的涨跌家数和成交额是
   `FROM daily_bar` 全表口径，ETF 现在日成交 2874 亿（个股 17952 亿），
   混进去"两市成交额"直接虚高 16%，而且以后谁新写一条 SQL 忘了过滤
   就是一次静默的统计错误。分表是唯一安全的做法。

2. **复权因子要转成"越大越新"的股票口径。** ETF 没有新浪 adj_factor，
   只能自己从分红算。自然算出来的是后向系数 f_t（末期=1、早期<1），
   而 `query.get_price` 的公式是 `qfq = close × factor / factor_last`，
   要求因子递增。所以存的是 `factor = f_t / f_0`（首日=1，末期>1），
   这样 qfq/hfq 两个公式一行都不用改。直接存 f_t 会让前复权反向缩放，
   图和收益全是反的。

3. **整只替换而不是增量合并。** 每来一次分红，全部历史行的因子都会变。
   增量写只补新 bar 的话，历史 adj_factor 会停在旧值，图上除权日出现
   假跳空。单只 1655 根、0.3 秒，整只重写最省心。
"""
from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.db import connect, exec_dml, init_schema, scalar
from src.sources.akshare_source import _retry

# 上交所 ETF 段与深交所 ETF 段。只用于"猜"，权威判断以 etf_dim 为准。
# 2026 年实际存在的段位：沪市 510/511/512/513/515/516/517/518/520/526/530/560/561/562/563/588/589，
# 深市 158/159。50 段现在已经没有 ETF（老的 500xxx 是 LOF）。
SH_PREFIX = ("51", "52", "53", "56", "58")
SZ_PREFIX = ("158", "159")

# 名称关键词 -> 分类。顺序有意义：先匹配到的优先，
# 所以"港股通消费ETF"必须先于"消费"命中跨境。
CATEGORY_RULES = [
    ("货币", ("货币", "日利", "添益", "保证金", "现金")),
    ("债券", ("债", "国债", "政金", "信用债", "短融")),
    ("商品", ("黄金", "白银", "豆粕", "原油", "有色期货", "商品")),
    ("跨境", ("港", "恒生", "H股", "纳指", "纳斯达克", "标普", "道琼斯",
              "日经", "德国", "法国", "东南亚", "沙特", "越南", "中韩", "亚太")),
    ("宽基", ("沪深300", "上证50", "中证500", "中证1000", "中证2000", "A500",
              "A50", "科创50", "科创100", "创业板", "深证100", "中证100",
              "上证180", "上证380", "全指", "MSCI")),
    ("行业", ("医药", "医疗", "白酒", "食品", "消费", "芯片", "半导体", "军工",
              "银行", "券商", "证券", "保险", "地产", "基建", "新能源", "光伏",
              "电池", "汽车", "有色", "煤炭", "钢铁", "电力", "化工", "传媒",
              "游戏", "通信", "计算机", "人工智能", "机器人", "农业", "养殖",
              "旅游", "红利", "价值", "成长", "低波", "家电", "粮食", "能源",
              "国企", "央企", "ESG", "基本面", "环保", "低碳", "质量", "稀土",
              "稀有", "建材", "机械", "物流", "航空", "港口", "教育", "养老")),
]

START_DATE = "2018-01-01"   # 早于多数 ETF 上市日；2018 前的老 ETF 也会被补齐


# ---------------------------------------------------------------- 代码
def to_ts_code(symbol: str) -> str:
    """6 位 ETF 代码 -> ts_code。股票版的 to_ts_code 把 510300 判成 .SZ，不能复用。"""
    s = str(symbol).zfill(6)
    if s.startswith(SH_PREFIX):
        return f"{s}.SH"
    if s.startswith(SZ_PREFIX):
        return f"{s}.SZ"
    # 其余 15/16/18 段是 LOF，也有交易所归属，但不在 ETF 范围
    return f"{s}.SZ" if s[0] == "1" else f"{s}.SH"


def to_sina(ts_code: str) -> str:
    """510300.SH -> sh510300"""
    code, _, ex = str(ts_code).partition(".")
    return ("sh" if ex.upper() == "SH" else "sz") + code


def is_etf_code(symbol: str) -> bool:
    """按代码段粗判，用于 resolve 时还没查库的场合。"""
    s = str(symbol).zfill(6)
    return s.startswith(SH_PREFIX) or s.startswith(SZ_PREFIX)


def classify(name: str) -> str:
    n = str(name or "")
    for label, kws in CATEGORY_RULES:
        if any(k in n for k in kws):
            return label
    return "其他"


# ---------------------------------------------------------------- 复权因子
def build_adj_factor(dates: pd.Series, closes: pd.Series,
                     divs: pd.DataFrame) -> np.ndarray:
    """按分红重建复权因子，返回"越大越新"的序列（首日=1）。

    dates 升序，长度 n；divs 需含 ex_date / amount（每份分红，非累计）。

    数学：自然算出来的是后向系数
        f_t = Π_{i: ex_i > t} (1 - div_i / pre_i)
    含义是"t 时点 1 份，在扣掉之后所有分红后相当于末日的多少份"，
    所以 f 递减、末期为 1。而 get_price 要的是递增因子，故归一：
        factor_t = f_t / f_0
    这样 qfq = close × factor/factor_last = close × f_t，正好是前复权价。
    """
    n = len(dates)
    if n == 0:
        return np.ones(0)

    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    c = pd.Series(closes).reset_index(drop=True).astype(float)

    # 每根 bar 当日的权益折损，默认 1（无事件）
    ratio = np.ones(n, dtype=float)

    # 注意：这里不能因为"没有分红"就提前返回。份额折算的检测不依赖分红表，
    # 一旦早退，107 只做过折算的宽基（中证1000 系列 2022 年）会全部漏掉，
    # 图上直接立一根 4 倍柱。
    if divs is not None and not divs.empty:
        for _, row in divs.iterrows():
            ex = pd.Timestamp(row["ex_date"])
            amt = float(row["amount"])
            if not np.isfinite(amt) or amt <= 0:
                continue
            # 除息日当天第一根 bar 的位置；pre 取它前一根的收盘
            j = int(d.searchsorted(ex))
            if j <= 0 or j >= n:
                # j==0：上市前就分红，没有可比的前收；j>=n：未来事件。都跳过。
                continue
            pre = float(c.iloc[j - 1])
            if not np.isfinite(pre) or pre <= 0:
                continue
            # 单次分红超过股价一半，几乎必然是份额拆分或脏数据。
            # 按分红处理会把整段历史价压成零头，而拆分本身不改变持有人权益。
            if amt / pre > 0.5:
                continue
            ratio[j] = 1.0 - amt / pre

    # 份额折算：东财在折算日仍按折算前的价格给 pre_close，于是 pct_chg 会显示
    # +295% 这种数（510580 于 2024-09-02 从 0.576 跳到 2.278，4 份合 1）。
    # 分红接口不覆盖这类事件，只能从价格跳变自己认。
    # 正常 ETF 没有涨跌停，但单日极少超过 50%，1.6 / 0.62 的边界是安全的；
    # 不处理的话 K 线上会立一根 4 倍柱，整张图的 y 轴被压扁。
    for i in range(1, n):
        try:
            prev, cur = float(c.iloc[i - 1]), float(c.iloc[i])
        except (TypeError, ValueError):
            continue
        if prev <= 0 or cur <= 0 or not np.isfinite(prev) or not np.isfinite(cur):
            continue
        r = cur / prev
        if r > 1.6 or r < 0.62:
            # 方向和分红相反：分红让历史价变小（乘 1-div/pre < 1），
            # 份额合并让历史价变大（1 份变 4 份，历史 1.0 元要显示成 4.0）。
            # 这里乘 r 而不是 1/r —— 写成 1/r 会让 qfq 变成 0.25，图上反向跳一个坑。
            ratio[i] *= r

    # 后向累乘：f[i] = ratio[i+1] * ratio[i+2] * ...
    f = np.ones(n, dtype=float)
    acc = 1.0
    for i in range(n - 2, -1, -1):
        acc *= ratio[i + 1]
        f[i] = acc
    f[-1] = 1.0

    base = f[0]
    return f / base if base > 0 else np.ones(n)


# ---------------------------------------------------------------- 抓取
def fetch_dim(verbose: bool = True) -> pd.DataFrame:
    """全市场 ETF 列表 + 当日快照。一次请求拿全量。"""
    s = _retry(_etf_spot)
    if s is None or s.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        "symbol": s["代码"].astype(str).str.zfill(6),
        "name": s["名称"].astype(str),
        "fund_size": pd.to_numeric(s.get("总市值"), errors="coerce"),
        "shares": pd.to_numeric(s.get("最新份额"), errors="coerce"),
    })
    df = df[df["symbol"].map(is_etf_code)].drop_duplicates("symbol")
    df["ts_code"] = df["symbol"].map(to_ts_code)
    df["exchange"] = df["ts_code"].str.split(".").str[1]
    df["category"] = df["name"].map(classify)
    df["updated_at"] = pd.Timestamp.now()

    idx = s.copy()
    idx["_code"] = idx["代码"].astype(str).str.zfill(6)
    idx = idx.drop_duplicates("_code").set_index("_code")
    sub = idx.reindex(df["symbol"].values)
    price = pd.to_numeric(sub["最新价"], errors="coerce")
    iopv = pd.to_numeric(sub["IOPV实时估值"], errors="coerce")
    # 东财的"基金折价率"正负号和直觉相反（负值=溢价）。这里一律用 价/IOPV 重算，
    # 正值就是溢价——否则 513100 这种 +17.5% 的溢价会被存成 -17.53，
    # 面板上"折价榜"和"溢价榜"会整个反过来。
    spot = pd.DataFrame({
        "ts_code": df["ts_code"].values,
        "trade_date": pd.Timestamp(_spot_date(s)),
        "price": price.values,
        "iopv": iopv.values,
        "premium": ((price / iopv - 1) * 100).values,
        "amount": pd.to_numeric(sub["成交额"], errors="coerce").values,
        "turnover": pd.to_numeric(sub["换手率"], errors="coerce").values,
        "shares": df["shares"].values,
        "fund_size": df["fund_size"].values,
    })

    con = connect()
    try:
        init_schema(con)
        _upsert_dim(con, df)
        _upsert_spot(con, spot)
    finally:
        con.close()
    if verbose:
        print(f"[ETF] 列表 {len(df)} 只，分类 " +
              str(df["category"].value_counts().to_dict()))
    return df


def _etf_spot() -> pd.DataFrame:
    """ETF 实时快照。优先自己并发翻页，失败再退回 akshare。

    东财这个接口把 pz 限死在 100 条（传 6000 也只回 100），1664 只要 17 页。
    akshare 的 `fund_etf_spot_em` 是串行翻页，实测 20.3 秒——放进页面交互里
    就是一按筛选转 20 秒。这里改成并发翻页，实测 1 秒内。
    """
    try:
        return _spot_fast()
    except Exception:
        import akshare as ak
        return ak.fund_etf_spot_em()


_SPOT_URL = "https://push2delay.eastmoney.com/api/qt/clist/get"
_SPOT_FS = "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827"
_SPOT_FIELDS = "f2,f3,f6,f8,f12,f14,f18,f20,f21,f38,f402,f441,f297,f124"
_SPOT_RENAME = {
    "f12": "代码", "f14": "名称", "f2": "最新价", "f3": "涨跌幅", "f6": "成交额",
    "f8": "换手率", "f18": "昨收", "f20": "总市值", "f21": "流通市值",
    "f38": "最新份额", "f402": "基金折价率", "f441": "IOPV实时估值",
    "f297": "数据日期", "f124": "更新时间",
}


def _spot_page(pn: int, pz: int = 100) -> tuple[int, list]:
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode({
        "pn": str(pn), "pz": str(pz), "po": "1", "np": "1", "fid": "f12",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": "2", "invt": "2",
        "wbp2u": "|0|0|0|web", "fs": _SPOT_FS, "fields": _SPOT_FIELDS,
    })
    req = urllib.request.Request(
        f"{_SPOT_URL}?{q}",
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
    )
    raw = urllib.request.urlopen(req, timeout=20).read()
    d = json.loads(raw.decode("utf-8", errors="replace"))
    data = d.get("data") or {}
    diff = data.get("diff") or []
    # 东财有时把 diff 给成 {"0": {...}, "1": {...}} 而不是数组
    if isinstance(diff, dict):
        diff = list(diff.values())
    return int(data.get("total") or 0), diff


def _spot_fast() -> pd.DataFrame:
    import math
    total, first = _spot_page(1)
    if not total or not first:
        raise RuntimeError("快照首页无数据")
    pages = math.ceil(total / 100)
    rows = list(first)
    if pages > 1:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(_spot_page, p): p for p in range(2, pages + 1)}
            for f in as_completed(futs):
                try:
                    rows.extend(f.result(timeout=30)[1])
                except Exception:
                    pass  # 单页失败就少几条，别让整张表陪葬
    df = pd.DataFrame(rows).rename(columns=_SPOT_RENAME)
    if "代码" not in df.columns:
        raise RuntimeError(f"字段名不匹配：{list(df.columns)[:6]}")
    # 调接口时拿到的数据日期，用于区分"今天收盘"和"昨天收盘"。
    # 原始是 YYYYMMDD 整数串，必须归一成 YYYY-MM-DD：_spot_date 用
    # pd.Timestamp(...[:10]) 解析，喂 "20261103" 会失败并静默退回 today()，
    # 于是周末跑一次就把周五的快照记成周六。
    if "数据日期" in df.columns:
        raw = df["数据日期"].astype(str).str.replace("-", "", regex=False).str[:8]
        parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        df["数据日期"] = parsed.dt.strftime("%Y-%m-%d")
    if "数据日期" not in df.columns or df["数据日期"].isna().all():
        df["数据日期"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    else:
        df["数据日期"] = df["数据日期"].fillna(
            pd.Timestamp.now().strftime("%Y-%m-%d"))
    return df.drop_duplicates("代码").reset_index(drop=True)


# 实时快照的节流状态。筛选页和 K 线页会交替读同一个价，两处差半小时就是
# "页面上两个数字不一样"这类最难查的问题，所以筛选前顺手刷一次快照。
_SPOT_LOCK = threading.Lock()
_SPOT_AT = 0.0


def refresh_spot(ttl: float = 45.0, force: bool = False) -> dict:
    """拉一次实时快照入库。ttl 秒内不重复请求。

    只更新 etf_spot（价格/IOPV/溢价/成交额/份额），不碰 etf_dim——
    分类和代码段不会因为盘中价格变化而变，重写 1664 行没必要。
    """
    global _SPOT_AT
    if not force and time.time() - _SPOT_AT < ttl:
        return {"刷新": False, "理由": "节流", "上次数秒前": round(time.time() - _SPOT_AT)}

    with _SPOT_LOCK:
        # 双重检查：并发请求里只让第一个真正去打接口
        if not force and time.time() - _SPOT_AT < ttl:
            return {"刷新": False, "理由": "节流"}
        s = _retry(_etf_spot)
        if s is None or s.empty:
            return {"刷新": False, "错误": "快照接口无数据，沿用上次"}

        con = connect()
        try:
            dim = con.execute("SELECT symbol, ts_code FROM etf_dim").df()
            if dim.empty:
                return {"刷新": False, "错误": "etf_dim 为空，先跑 fetch_dim"}
            idx = s.copy()
            idx["_code"] = idx["代码"].astype(str).str.zfill(6)
            idx = idx.drop_duplicates("_code").set_index("_code")
            sub = idx.reindex(dim["symbol"].values)
            price = pd.to_numeric(sub["最新价"], errors="coerce")
            iopv = pd.to_numeric(sub["IOPV实时估值"], errors="coerce")
            spot = pd.DataFrame({
                "ts_code": dim["ts_code"].values,
                "trade_date": pd.Timestamp(_spot_date(s)),
                "price": price.values,
                "iopv": iopv.values,
                "premium": ((price / iopv - 1) * 100).values,
                "amount": pd.to_numeric(sub["成交额"], errors="coerce").values,
                "turnover": pd.to_numeric(sub["换手率"], errors="coerce").values,
                "shares": pd.to_numeric(sub["最新份额"], errors="coerce").values,
                "fund_size": pd.to_numeric(sub["总市值"], errors="coerce").values,
            })
            _upsert_spot(con, spot)
            n = len(spot)
        finally:
            con.close()
        _SPOT_AT = time.time()
        return {"刷新": True, "条数": int(n)}


def _spot_date(s: pd.DataFrame) -> object:
    """快照自带数据日期，别用 today（晚上跑会被记成第二天）。"""
    for col in ("数据日期", "更新时间"):
        if col in s.columns and len(s):
            try:
                return pd.Timestamp(str(s[col].iloc[0])[:10]).date()
            except Exception:
                continue
    return pd.Timestamp.now().date()


def _upsert_dim(con, df: pd.DataFrame) -> None:
    if df.empty:
        return
    con.register("_dim", df[["ts_code", "symbol", "name", "exchange",
                             "category", "fund_size", "shares", "updated_at"]])
    try:
        exec_dml(con, """
            INSERT INTO etf_dim
            SELECT * FROM _dim
            ON CONFLICT (ts_code) DO UPDATE SET
                name=excluded.name, exchange=excluded.exchange,
                category=excluded.category, fund_size=excluded.fund_size,
                shares=excluded.shares, updated_at=excluded.updated_at
        """)
    finally:
        con.unregister("_dim")


def _upsert_spot(con, df: pd.DataFrame) -> None:
    if df.empty:
        return
    con.register("_spot", df)
    try:
        exec_dml(con, """
            INSERT INTO etf_spot
            SELECT * FROM _spot
            ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                price=excluded.price, iopv=excluded.iopv, premium=excluded.premium,
                amount=excluded.amount, turnover=excluded.turnover,
                shares=excluded.shares, fund_size=excluded.fund_size
        """)
    finally:
        con.unregister("_spot")


def fetch_div(ts_code: str) -> pd.DataFrame:
    """单只 ETF 的分红明细（每份金额）。取不到返回空表。"""
    import akshare as ak
    try:
        d = ak.fund_etf_dividend_sina(symbol=to_sina(ts_code))
    except Exception:
        return pd.DataFrame(columns=["ex_date", "amount"])
    if d is None or d.empty:
        return pd.DataFrame(columns=["ex_date", "amount"])
    d = d.copy()
    d["ex_date"] = pd.to_datetime(d["日期"])
    d = d.sort_values("ex_date").reset_index(drop=True)
    # 接口给的是累计分红，差分才是单次；第一条没有前值，本身就不是单次分红，
    # 直接丢掉（它会被当成"上市以来全部分红"，把因子拉爆）
    d["amount"] = d["累计分红"].diff()
    d = d[d["amount"].notna() & (d["amount"] > 0)]
    return d[["ex_date", "amount"]]


def fetch_bars(ts_code: str, start: str = START_DATE) -> pd.DataFrame:
    """不复权日线。复权在写入前用分红算好一起写。"""
    import akshare as ak
    sym = str(ts_code).split(".")[0]
    try:
        raw = _retry(ak.fund_etf_hist_em, symbol=sym, period="daily",
                     start_date=start.replace("-", ""),
                     end_date=pd.Timestamp.now().strftime("%Y%m%d"), adjust="")
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.rename(columns={
        "日期": "trade_date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "vol", "成交额": "amount", "振幅": "amplitude",
        "涨跌幅": "pct_chg", "涨跌额": "price_chg", "换手率": "turnover",
    })
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in ("open", "high", "low", "close", "vol", "amount",
              "amplitude", "pct_chg", "price_chg", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None
    df["pre_close"] = df["close"] - df["price_chg"]
    df["ts_code"] = ts_code

    # 盘中不落当天这根。东财 hist 在 15:00 前返回的是"此刻"的价格，
    # 存进库就会被当成收盘价：kline 看到库里已有今天的数据，就不再拼实时 bar，
    # 结果盘中看图一直停在抓取那一刻的价，越看越不动。
    # 当天这根交给 kline._live_bar 用实时报价拼，收盘后下一次更新再补进来。
    now = pd.Timestamp.now()
    today = now.normalize()
    if now.hour * 60 + now.minute < 15 * 60:
        df = df[df["trade_date"] < today]
    return df.sort_values("trade_date").reset_index(drop=True)


def save_bars(con, ts_code: str, bars: pd.DataFrame,
              divs: pd.DataFrame) -> int:
    """整只替换：先删后插，因子按最新分红全量重算。"""
    if bars is None or bars.empty:
        return 0
    b = bars.copy()
    b["adj_factor"] = build_adj_factor(b["trade_date"], b["close"], divs)
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "vol", "amount", "amplitude", "pct_chg", "price_chg", "turnover",
            "adj_factor"]
    b = b[cols]
    exec_dml(con, "DELETE FROM etf_bar WHERE ts_code = ?", [ts_code])
    con.register("_b", b)
    try:
        exec_dml(con, "INSERT INTO etf_bar SELECT * FROM _b")
    finally:
        con.unregister("_b")
    return len(b)


# ---------------------------------------------------------------- 更新
def update(codes: list[str] | None = None, workers: int = 3,
           with_div: bool = True, limit: int | None = None,
           verbose: bool = True) -> dict:
    """抓取 ETF 日线（含复权因子）。codes 为空则更新全市场。

    workers 默认 3：东财在 8 并发时会返回 HTTP 567 风控页（公告接口实测），
    基金接口更保守一些更稳。
    """
    if codes:
        todo = [c if "." in c else to_ts_code(c) for c in codes]
    else:
        con = connect(read_only=True)
        try:
            todo = [r[0] for r in con.execute(
                "SELECT ts_code FROM etf_dim ORDER BY fund_size DESC NULLS LAST"
            ).fetchall()]
        finally:
            con.close()
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"总数": 0, "成功": 0, "空": 0, "失败": 0}

    t0 = time.time()
    divs = {}
    if with_div:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_div, c): c for c in todo}
            for f in as_completed(futs):
                c = futs[f]
                try:
                    divs[c] = f.result()
                except Exception:
                    divs[c] = pd.DataFrame(columns=["ex_date", "amount"])

    ok = empty = fail = 0
    failed: list[str] = []
    con = connect()
    try:
        init_schema(con)

        def _one(code):
            bars = fetch_bars(code)
            if bars.empty:
                return code, None, None
            return code, bars, divs.get(code, pd.DataFrame(columns=["ex_date", "amount"]))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one, c): c for c in todo}
            done = 0
            for f in as_completed(futs):
                code = futs[f]
                done += 1
                try:
                    code, bars, dv = f.result()
                except Exception:
                    fail += 1
                    failed.append(code)
                    continue
                if bars is None:
                    empty += 1
                    continue
                try:
                    save_bars(con, code, bars, dv)
                    ok += 1
                except Exception:
                    fail += 1
                    failed.append(code)
                if verbose and done % 100 == 0:
                    print(f"  {done}/{len(todo)}  成功 {ok}  空 {empty}  失败 {fail}")
    finally:
        con.close()

    res = {"总数": len(todo), "成功": ok, "空": empty, "失败": fail,
           "耗时秒": round(time.time() - t0, 1), "失败样例": failed[:8]}
    if verbose:
        print(f"[ETF] 更新完成 {res}")
    return res


# ---------------------------------------------------------------- 筛选
# 个股评分器的六个维度（ROE / 净利同比 / 毛利率 / 净利 / 动能 / 估值分位）里，
# ETF 只有"动能"一项有定义：没有 ROE、没有毛利率、没有 PE 分位。硬套的结果
# 是评分退化成"近 20 日涨幅榜"，还挂着"选股"的名义。
#
# ETF 真正要看的四项，全部是它特有的：
#   - 规模：<5000 万有清盘风险，1 亿以下流动性通常也不行
#   - 流动性：日均成交额决定了买卖的冲击成本，比当日成交额稳
#   - 折溢价：溢价买入等于当场亏，跨境 ETF 尤其容易失控
#   - 跟踪/涨跌：比的是同类里谁跑得动，不是绝对值
_SORT_LABEL = {
    "amt": "20日日均成交额（流动性优先）",
    "amount": "今日成交额",
    "size": "规模",
    "premium": "溢价率（高→低）",
    "discount": "折价率（低→高）",
    "ret20": "20日涨幅（高→低）",
    "ret20_asc": "20日涨幅（低→高，找跌得多的）",
    "turnover": "换手率",
    "code": "代码",
}

_SCREEN_SQL = """
WITH b AS (
    SELECT ts_code, trade_date, close, amount, adj_factor,
           close * adj_factor AS adjc,
           row_number() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) AS rn,
           count(*)   OVER (PARTITION BY ts_code) AS n
    FROM etf_bar
), a AS (
    SELECT ts_code,
           max(CASE WHEN rn = 1  THEN adjc  END) AS adjc_0,
           max(CASE WHEN rn = 1  THEN close END) AS close_0,
           max(CASE WHEN rn = 2  THEN adjc  END) AS adjc_1,
           max(CASE WHEN rn = 20 THEN adjc  END) AS adjc_19,
           max(CASE WHEN rn = 21 THEN adjc  END) AS adjc_20,
           max(CASE WHEN rn = 60 THEN adjc  END) AS adjc_59,
           max(CASE WHEN rn = 61 THEN adjc  END) AS adjc_60,
           max(CASE WHEN rn = 1  THEN trade_date END) AS bar_date,
           avg(CASE WHEN rn <= 20 THEN amount END) AS amt20,
           max(n) AS n_bars
    FROM b WHERE rn <= 61 GROUP BY ts_code
)
SELECT d.ts_code, d.name, d.category, d.exchange,
       s.price, s.iopv, s.premium, s.amount, s.turnover,
       s.shares, s.fund_size, s.trade_date AS spot_date,
       a.close_0, a.bar_date, a.n_bars, a.amt20,
       a.adjc_0, a.adjc_1, a.adjc_19, a.adjc_20, a.adjc_59, a.adjc_60
FROM etf_dim d
JOIN a USING (ts_code)
LEFT JOIN etf_spot s
       ON s.ts_code = d.ts_code
      AND s.trade_date = (SELECT max(trade_date) FROM etf_spot)
"""


def _pct(a, b):
    """b 缺失或不正时返回 None——新上市基金算不出 20 日涨幅，不能给 0。"""
    try:
        a, b = float(a), float(b)
        if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
            return None
        return round((a / b - 1) * 100, 2)
    except (TypeError, ValueError):
        return None


def _f(v, nd=2):
    try:
        v = float(v)
        return round(v, nd) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def screen(category: str | None = None,
           kw: str | None = None,
           min_amount: float = 500.0,
           min_size: float = 0.0,
           max_premium: float | None = None,
           min_days: int = 0,
           exclude: tuple | list | None = ("货币", "债券"),
           sort: str = "amt",
           limit: int = 60,
           refresh: bool = False) -> dict:
    """按 ETF 自己的口径筛，不复用个股评分。

    min_amount 单位万元（20 日日均成交额），min_size 单位亿元，
    max_premium 单位 %。默认门槛是"日均 500 万 + 不限规模"，
    因为低于 500 万的 ETF 买卖一万块就能把价格打歪。

    `exclude` 默认剔掉货币和债券：这两类是现金管理工具，年化就 2% 左右，
    涨跌幅常年贴着 0，却因为盘子大、换手高，必然霸占"流动性优先"的前几名
    ——把它们留在默认榜里，第一屏永远是两只收益为零的货基。传 None 看全部。

    `refresh=True` 会先刷一次实时快照（TTL 45 秒），让筛选页和 K 线页
    看到的是同一个时点的价。刷新失败不阻断筛选，沿用库里的旧快照。
    """
    spot_note = None
    if refresh:
        try:
            spot_note = refresh_spot()
        except Exception as e:  # 写连接被其他 read_only 连接挡住等
            spot_note = {"刷新": False, "错误": f"{type(e).__name__}: {str(e)[:80]}"}

    con = connect(read_only=True)
    try:
        df = con.execute(_SCREEN_SQL).df()
    finally:
        con.close()

    total = len(df)
    if df.empty:
        return {"总数": 0, "命中": 0, "说明": "ETF 库为空，先跑 python -m src.etf update"}

    # spot 与 bar 常常不同步：盘中 bar 只到昨天收盘，spot 是今天的实时价。
    # 若直接"用 bar 收盘算涨幅、用 spot 价显示现价"，盘中会出现
    # "现价一直在跌、20 日涨幅却纹丝不动"。统一折算到同一复权基准：
    #   f = adjc_0 / close_0（逐日恒定），最新复权价 = 最新价 × f
    close0 = pd.to_numeric(df["close_0"], errors="coerce")
    live = pd.to_numeric(df["price"], errors="coerce")
    df["f"] = np.where(close0 > 0, pd.to_numeric(df["adjc_0"], errors="coerce") / close0, np.nan)
    newer = live.notna() & (live > 0) & (
        pd.to_datetime(df["spot_date"], errors="coerce")
        > pd.to_datetime(df["bar_date"], errors="coerce")
    )
    df["px"] = np.where(newer, live, close0)
    df["adjc_now"] = df["px"] * df["f"]

    # spot 比 bar 新一根时，等价于序列尾部多拼了今天这根，所有"N 日"基准
    # 都要整体前移一位。直接用 adjc_1 会把"今日涨跌"算成"昨日涨跌"——
    # 实测 588000 今天跌 4.84%，用旧基准却显示 +2.61%，方向反了。
    b1 = np.where(newer, df["adjc_0"], df["adjc_1"])
    b20 = np.where(newer, df["adjc_19"], df["adjc_20"])
    b60 = np.where(newer, df["adjc_59"], df["adjc_60"])

    df["涨幅1"] = [_pct(x, y) for x, y in zip(df["adjc_now"], b1)]
    df["涨幅20"] = [_pct(x, y) for x, y in zip(df["adjc_now"], b20)]
    df["涨幅60"] = [_pct(x, y) for x, y in zip(df["adjc_now"], b60)]
    df["规模亿"] = pd.to_numeric(df["fund_size"], errors="coerce") / 1e8
    df["成交额亿"] = pd.to_numeric(df["amount"], errors="coerce") / 1e8
    df["日均成交万"] = pd.to_numeric(df["amt20"], errors="coerce") / 1e4
    df["溢价"] = pd.to_numeric(df["premium"], errors="coerce")
    df["换手"] = pd.to_numeric(df["turnover"], errors="coerce")

    if exclude:
        df = df[~df["category"].isin(list(exclude))]
    if category and category not in ("全部", ""):
        df = df[df["category"] == category]
    if kw:
        k = "".join(str(kw).split())
        mask = (df["name"].fillna("").str.replace(" ", "", regex=False).str.contains(k, na=False)
                | df["ts_code"].fillna("").str.contains(k, na=False))
        df = df[mask]

    # 流动性门槛只作用于"有成交"的池子：预留了一批日成交 0 的僵尸 ETF，
    # 它们本来就买不进去，留在列表里只会把真正的候选挤下去。
    if min_amount and min_amount > 0:
        df = df[df["日均成交万"].fillna(0) >= float(min_amount)]
    if min_size and min_size > 0:
        df = df[df["规模亿"].fillna(0) >= float(min_size)]
    if max_premium is not None:
        # 溢价超过阈值的剔除；溢价率缺失（无 IOPV 的品种）保留，不能因缺数据误杀
        df = df[df["溢价"].isna() | (df["溢价"] <= float(max_premium))]
    if min_days and min_days > 0:
        df = df[df["n_bars"].fillna(0) >= int(min_days)]

    n_filt = len(df)

    key = sort if sort in _SORT_LABEL else "amt"
    col = {"amt": "日均成交万", "amount": "成交额亿", "size": "规模亿",
           "premium": "溢价", "discount": "溢价", "ret20": "涨幅20",
           "ret20_asc": "涨幅20", "turnover": "换手", "code": "ts_code"}[key]
    asc = key in ("discount", "ret20_asc", "code")
    df = df.sort_values(col, ascending=asc, na_position="last")

    limit = max(1, min(int(limit or 60), 500))
    rows = []
    for _, r in df.head(limit).iterrows():
        rows.append({
            "代码": r["ts_code"],
            "名称": str(r["name"] or "").replace(" ", ""),
            "分类": r["category"],
            "现价": _f(r["px"], 3),
            "今日%": _f(r["涨幅1"]),
            "溢价%": _f(r["溢价"]),
            "成交额亿": _f(r["成交额亿"]),
            "规模亿": _f(r["规模亿"], 1),
            "日均成交万": _f(r["日均成交万"], 0),
            "20日%": _f(r["涨幅20"]),
            "60日%": _f(r["涨幅60"]),
            "换手%": _f(r["换手"]),
            "根数": int(r["n_bars"] or 0),
            "快照日": str(r["spot_date"])[:10] if pd.notna(r["spot_date"]) else None,
        })

    return {
        "总数": total,
        "命中": n_filt,
        "返回": len(rows),
        "排序": _SORT_LABEL[key],
        "快照": spot_note,
        "条件": {
            "分类": category or "全部",
            "排除": list(exclude) if exclude else [],
            "关键词": kw or None,
            "日均成交额≥万": min_amount,
            "规模≥亿": min_size,
            "溢价≤%": max_premium,
        },
        "行": rows,
    }


def premium_board(limit: int = 15, min_amount: float = 500.0) -> dict:
    """折溢价两端。溢价买入当场就亏这部分，是 ETF 唯一"看一眼就能亏钱"的指标。"""
    hi = screen(min_amount=min_amount, sort="premium", limit=limit)
    lo = screen(min_amount=min_amount, sort="discount", limit=limit)
    return {
        "溢价榜": hi["行"],
        "折价榜": lo["行"],
        "门槛": f"20 日均成交额 ≥ {min_amount} 万",
        "说明": ("溢价率 =(现价 / IOPV − 1)×100，正值是溢价。"
                 "跨境 ETF 因外汇额度限制常年高溢价，溢价买入意味着"
                 "即使指数不动也在亏；折价则相反。"),
    }


def stats() -> dict:
    con = connect(read_only=True)
    try:
        dim = scalar(con, "SELECT count(*) FROM etf_dim") or 0
        row = con.execute(
            "SELECT count(*), count(DISTINCT ts_code), min(trade_date), max(trade_date) "
            "FROM etf_bar").fetchone()
        spot = scalar(con, "SELECT count(*) FROM etf_spot") or 0
        cat = con.execute(
            "SELECT category, count(*) FROM etf_dim GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        return {"ETF数": int(dim), "日线行数": int(row[0] or 0),
                "覆盖只数": int(row[1] or 0),
                "起": str(row[2]) if row[2] else None,
                "止": str(row[3]) if row[3] else None,
                "快照行数": int(spot),
                "分类": {r[0]: r[1] for r in cat}}
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "dim":
        fetch_dim()
    elif cmd == "update":
        update(sys.argv[2:] or None)
    else:
        print(stats())
