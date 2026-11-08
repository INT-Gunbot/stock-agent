"""初始化数据库：建表 + 股票列表 + 指数基础数据。

用法:  python -m scripts.init_db
"""
import sys

from src.db import connect, init_schema
from src.update import update_index, update_stock_list


def main():
    con = connect()
    init_schema(con)
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1"
    ).fetchall()]
    print(f"[建表] 完成，共 {len(tables)} 张表: {', '.join(tables)}")
    con.close()

    update_stock_list()
    update_index(full=True)
    print("[完成] 数据库初始化结束")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
