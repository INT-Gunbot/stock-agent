"""一次性修复：给 adj_factor 表补回主键约束。

背景：`update_factors` 曾用 `CREATE TABLE AS SELECT` 整表重建，CTAS 不携带约束，
导致 adj_factor 变成一张没有任何主键的裸表。数据本身没坏（行数与去重后一致），
但后果是 `INSERT ... ON CONFLICT` 直接报 Binder Exception，且今后一旦写入逻辑
出岔子，重复行会静默入库，所有 join adj_factor 的地方都会被放大成笛卡尔积。

用法：
    python -m scripts.fix_adj_factor_pk          # 检查并修复
    python -m scripts.fix_adj_factor_pk --check  # 只检查

注意：DuckDB 单文件写入独占，运行前要先停掉 demo 服务。
"""
from __future__ import annotations

import argparse
import sys

from src.db import connect, exec_dml, scalar


def has_pk(con, table: str = "adj_factor") -> bool:
    rows = con.execute(
        "SELECT constraint_type FROM duckdb_constraints() WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
        [table],
    ).fetchall()
    return bool(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查，不修改")
    args = ap.parse_args(argv)

    con = connect()
    try:
        if has_pk(con):
            print("[修复] adj_factor 已有主键，无需处理")
            return 0

        n = scalar(con, "SELECT count(*) FROM adj_factor")
        n_distinct = scalar(con, "SELECT count(*) FROM (SELECT DISTINCT ts_code, trade_date FROM adj_factor)")
        print(f"[修复] adj_factor 共 {n:,} 行，去重后 {n_distinct:,} 行")
        if args.check:
            print("[修复] 缺少主键约束（--check 模式，未修改）")
            return 1

        exec_dml(con, "DROP TABLE IF EXISTS adj_factor_new")
        exec_dml(con, """
            CREATE TABLE adj_factor_new (
                ts_code     VARCHAR NOT NULL,
                trade_date  DATE NOT NULL,
                adj_factor  DOUBLE,
                PRIMARY KEY (ts_code, trade_date)
            )
        """)
        # 同一 (票, 日) 保留一行；正常数据无重复，这里只是防御
        exec_dml(con, """
            INSERT INTO adj_factor_new
            SELECT ts_code, trade_date, any_value(adj_factor)
            FROM adj_factor GROUP BY ts_code, trade_date
        """)
        n_new = scalar(con, "SELECT count(*) FROM adj_factor_new")
        if n_new != n_distinct:
            print(f"[修复] 行数不符（新表 {n_new} vs 期望 {n_distinct}），已回滚")
            exec_dml(con, "DROP TABLE adj_factor_new")
            return 2

        exec_dml(con, "DROP TABLE IF EXISTS adj_factor_old")
        exec_dml(con, "ALTER TABLE adj_factor RENAME TO adj_factor_old")
        exec_dml(con, "ALTER TABLE adj_factor_new RENAME TO adj_factor")
        exec_dml(con, "DROP TABLE adj_factor_old")

        ok = has_pk(con)
        print(f"[修复] 完成：{n_new:,} 行，主键 {'已建立' if ok else '建立失败'}")
        return 0 if ok else 3
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
