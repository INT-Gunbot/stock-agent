"""DuckDB 连接与建表。

设计要点（决定了能否正确回测）：
1. daily_bar 只存不复权原始价，复权在查询时动态计算 -> 避免"前复权价随分红漂移"
2. adj_factor 单独存 -> 复权可重算
3. fina_yjbb 用 (ts_code, end_date, ingested_at) 做主键 -> 财报追溯调整时保留多版本
4. 所有财务记录带 ann_date -> 回测按公告日过滤，避免前视偏差
"""
import duckdb

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_stock (
    ts_code     VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    name        VARCHAR,
    exchange    VARCHAR,
    industry    VARCHAR,
    is_active   BOOLEAN DEFAULT TRUE,
    updated_at  TIMESTAMP,
    PRIMARY KEY (ts_code)
);

CREATE TABLE IF NOT EXISTS daily_bar (
    ts_code     VARCHAR NOT NULL,
    trade_date  DATE NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    pre_close   DOUBLE,
    vol         DOUBLE,
    amount      DOUBLE,
    amplitude   DOUBLE,
    pct_chg     DOUBLE,
    price_chg   DOUBLE,
    turnover    DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS adj_factor (
    ts_code     VARCHAR NOT NULL,
    trade_date  DATE NOT NULL,
    adj_factor  DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS index_bar (
    index_code  VARCHAR NOT NULL,
    trade_date  DATE NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    vol         DOUBLE,
    amount      DOUBLE,
    PRIMARY KEY (index_code, trade_date)
);

CREATE TABLE IF NOT EXISTS fina_yjbb (
    ts_code        VARCHAR NOT NULL,
    end_date       DATE NOT NULL,
    ann_date       DATE,
    visible_date   DATE,
    ingested_at    DATE NOT NULL,
    eps            DOUBLE,
    revenue        DOUBLE,
    revenue_yoy    DOUBLE,
    revenue_qoq    DOUBLE,
    net_profit     DOUBLE,
    net_profit_yoy DOUBLE,
    net_profit_qoq DOUBLE,
    bps            DOUBLE,
    roe            DOUBLE,
    ocf_ps         DOUBLE,
    gross_margin   DOUBLE,
    industry       VARCHAR,
    PRIMARY KEY (ts_code, end_date, ingested_at)
);

CREATE TABLE IF NOT EXISTS market_snapshot (
    trade_date  DATE NOT NULL,
    metric      VARCHAR NOT NULL,
    value       DOUBLE,
    extra       VARCHAR,
    computed_at TIMESTAMP,
    PRIMARY KEY (trade_date, metric)
);

CREATE TABLE IF NOT EXISTS etf_dim (
    ts_code     VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    name        VARCHAR,
    exchange    VARCHAR,
    category    VARCHAR,
    fund_size   DOUBLE,
    shares      DOUBLE,
    updated_at  TIMESTAMP,
    PRIMARY KEY (ts_code)
);

CREATE TABLE IF NOT EXISTS etf_bar (
    ts_code     VARCHAR NOT NULL,
    trade_date  DATE NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    pre_close   DOUBLE,
    vol         DOUBLE,
    amount      DOUBLE,
    amplitude   DOUBLE,
    pct_chg     DOUBLE,
    price_chg   DOUBLE,
    turnover    DOUBLE,
    adj_factor  DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS etf_spot (
    ts_code     VARCHAR NOT NULL,
    trade_date  DATE NOT NULL,
    price       DOUBLE,
    iopv        DOUBLE,
    premium     DOUBLE,
    amount      DOUBLE,
    turnover    DOUBLE,
    shares      DOUBLE,
    fund_size   DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS update_log (
    task        VARCHAR,
    target      VARCHAR,
    status      VARCHAR,
    rows        BIGINT,
    message     VARCHAR,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP
);
"""


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection | None = None) -> None:
    own = con is None
    con = con or connect()
    try:
        con.execute(SCHEMA)
    finally:
        if own:
            con.close()


def exec_dml(con, sql: str, params: list | None = None) -> list:
    """执行 DDL/DML 并显式消费结果。

    DuckDB 的 connection 若存在未消费的结果集，下一次 execute 会抛
    "Resource In Use Error"。所有写操作统一走这里。
    """
    cur = con.execute(sql, params) if params else con.execute(sql)
    try:
        return cur.fetchall()
    except Exception:
        return []


def scalar(con, sql: str, params: list | None = None):
    """取首行首列。

    注意：DuckDB 2.0 中 fetchone() 取完唯一一行后结果集仍可能处于未完成状态，
    后续 register/unregister/execute 会抛 ResourceInUse，因此统一用 fetchall。
    """
    cur = con.execute(sql, params) if params else con.execute(sql)
    rows = cur.fetchall()
    return rows[0][0] if rows else None


def upsert_anti_join(con, df, table: str, keys: list[str]) -> int:
    """把 DataFrame 中库里不存在的行插入（按 keys 去重），返回新增行数。"""
    if df is None or len(df) == 0:
        return 0
    con.register("_staging_df", df)
    try:
        key_clause = " AND ".join(f"t.{k} = d.{k}" for k in keys)
        before = scalar(con, f"SELECT count(*) FROM {table}")
        exec_dml(
            con,
            f"INSERT INTO {table} SELECT t.* FROM _staging_df t "
            f"LEFT JOIN {table} d ON {key_clause} WHERE d.{keys[0]} IS NULL",
        )
        after = scalar(con, f"SELECT count(*) FROM {table}")
        return int(after - before)
    finally:
        con.unregister("_staging_df")
