"""数据质量检查：改数据层后必跑的回归测试。

用法:  python -m scripts.quality_check
"""
import sys

import pandas as pd

from src.db import connect


def check_adj_consistency(con) -> list[str]:
    """复权因子健康度。

    因子已换成新浪阶梯因子，"单日跳变"就是除权日本身（比亚迪 10 送 10 跳 203%、
    诺瓦星云 10 转 9 跳 84%），属于正确行为，不能再当异常报——否则这个检查
    每天都会亮红灯，最后没人看。

    真正需要报警的是两类：
      1. 因子为空或非正（会让 adj_close 变 NaN 或符号翻转）
      2. 因子出现"非阶梯"漂移（唯一值/交易日 > 30%），说明该票是东财兜底因子，
         技术指标精度下降，不应参与选股打分
      3. 因子下降超 5%（分红送转必然让因子上升，下降意味着缩股或数据错误）
    """
    issues = []

    rows = con.execute(
        """
        SELECT count(*) FILTER (WHERE adj_factor IS NULL) AS n_null,
               count(*) FILTER (WHERE adj_factor <= 0) AS n_bad
        FROM adj_factor
        """
    ).fetchall()[0]
    if rows[0]:
        issues.append(f"[因子为空] {rows[0]} 行（adj_close 会变 NaN）")
    if rows[1]:
        issues.append(f"[因子非正] {rows[1]} 行（收益率符号会翻转）")

    drift = con.execute(
        """
        WITH x AS (
            SELECT ts_code, count(*) AS n, count(DISTINCT adj_factor) AS u
            FROM adj_factor GROUP BY 1
        )
        SELECT count(*) AS n_drift FROM x WHERE u > greatest(3, n * 0.3)
        """
    ).fetchall()[0][0]
    if drift:
        issues.append(
            f"[因子非阶梯] {drift} 只（东财兜底因子，逐日漂移，技术指标不可用于选股）"
        )

    down = con.execute(
        """
        WITH x AS (
            SELECT ts_code, trade_date, adj_factor,
                   lag(adj_factor) OVER (PARTITION BY ts_code ORDER BY trade_date) AS p
            FROM adj_factor
        )
        SELECT ts_code, trade_date, round((adj_factor/p-1)*100, 2) AS chg
        FROM x WHERE p IS NOT NULL AND adj_factor / p - 1 < -0.05
        ORDER BY chg LIMIT 5
        """
    ).df()
    for _, r in down.iterrows():
        issues.append(
            f"[因子下降] {r['ts_code']} {str(r['trade_date'])[:10]} {r['chg']}%"
            f"（分红送转应使因子上升，需核对是否缩股）"
        )
    return issues


def check_price_sanity(con) -> list[str]:
    """用「修正后真实涨跌幅」判断，而不是原始 pct_chg。

    原始 pct_chg 在除权日必然超限（10 送 10 显示 -50%），
    比亚迪 2025-07-29 送转显示 -66.94%、科思股份 -49.73% 都是正常除权。
    真实异常 = 原始涨跌幅超限，但原始与复权收益率之差很小（说明不是除权）。
    """
    issues = []
    df = con.execute(
        """
        WITH orderday AS (
            SELECT ts_code, trade_date,
                   row_number() OVER (PARTITION BY ts_code ORDER BY trade_date) AS rn
            FROM daily_bar
        ),
        x AS (
            SELECT b.ts_code, b.trade_date, b.close, b.pre_close, b.pct_chg,
                   (b.close / b.pre_close - 1) AS raw_ret,
                   o.rn,
                   (b.close * a.adj_factor) AS adjc,
                   lag(b.close * a.adj_factor) OVER (
                       PARTITION BY b.ts_code ORDER BY b.trade_date) AS prev_adjc
            FROM daily_bar b
            JOIN adj_factor a USING (ts_code, trade_date)
            JOIN orderday o USING (ts_code, trade_date)
        )
        SELECT x.ts_code, x.trade_date, x.pct_chg, d.name,
               round((x.adjc / x.prev_adjc - 1 - x.raw_ret) * 100, 2) AS gap_pct
        FROM x
        LEFT JOIN dim_stock d ON d.ts_code = x.ts_code
        WHERE x.rn > 5
          AND x.ts_code NOT LIKE '%.BJ'
          AND abs(x.pct_chg) > 21
          AND abs(x.adjc / x.prev_adjc - 1 - x.raw_ret) < 0.03
          AND coalesce(d.name, '') NOT LIKE '%退%'
          AND coalesce(d.name, '') NOT LIKE '%ST%'
        ORDER BY abs(x.pct_chg) DESC LIMIT 20
        """
    ).df()
    for _, r in df.iterrows():
        issues.append(
            f"[涨跌幅异常] {r['ts_code']} {str(r['trade_date'])[:10]} "
            f"{r['pct_chg']}% ({r['name']}) 非除权，gap={r['gap_pct']}%"
        )
    return issues


def check_ohlc_logic(con) -> list[str]:
    """OHLC 逻辑：high 应 >= max(open, close)，low 应 <= min(open, close)。"""
    n = con.execute(
        """
        SELECT count(*) FROM daily_bar
        WHERE high < open OR high < close OR low > open OR low > close
           OR close <= 0 OR vol < 0
        """
    ).fetchall()[0][0]
    return [f"[OHLC 逻辑错误] {n} 行"] if n else []


def check_fina_pit(con) -> list[str]:
    """财务数据的可见性字段必须完整。"""
    issues = []
    n_null = con.execute(
        "SELECT count(*) FROM fina_yjbb WHERE visible_date IS NULL"
    ).fetchall()[0][0]
    if n_null:
        issues.append(f"[财务缺可见日] {n_null} 行（PIT 过滤会失效）")
    n_early = con.execute(
        "SELECT count(*) FROM fina_yjbb WHERE visible_date < end_date"
    ).fetchall()[0][0]
    if n_early:
        issues.append(f"[可见日早于报告期] {n_early} 行")
    n_cross = con.execute(
        """
        SELECT count(*) FROM fina_yjbb f
        LEFT JOIN dim_stock d USING(ts_code) WHERE d.ts_code IS NULL
        """
    ).fetchall()[0][0]
    if n_cross:
        issues.append(f"[财务混入非A股标的] {n_cross} 行")
    return issues


def check_coverage(con) -> pd.DataFrame:
    """覆盖率：多少只票、日期范围、最新交易日。"""
    return con.execute(
        """
        SELECT count(DISTINCT ts_code) AS stocks, count(*) AS rows,
               min(trade_date) AS first_day, max(trade_date) AS last_day,
               count(DISTINCT trade_date) AS trade_days
        FROM daily_bar
        """
    ).df()


def check_gaps(con, ts_code: str) -> pd.DataFrame:
    """检查单只票是否有日期缺口（对比全市场交易日历）。"""
    return con.execute(
        """
        WITH cal AS (SELECT DISTINCT trade_date FROM daily_bar),
             me AS (SELECT trade_date FROM daily_bar WHERE ts_code = ?)
        SELECT cal.trade_date FROM cal
        LEFT JOIN me ON me.trade_date = cal.trade_date
        WHERE me.trade_date IS NULL ORDER BY 1
        """,
        [ts_code],
    ).df()


def main():
    con = connect(read_only=True)
    print("=== 数据覆盖 ===")
    print(check_coverage(con).to_string(index=False))

    issues = []
    issues += check_ohlc_logic(con)
    issues += check_fina_pit(con)
    issues += check_adj_consistency(con)
    issues += check_price_sanity(con)

    print()
    if issues:
        print(f"=== 发现 {len(issues)} 个问题 ===")
        for i in issues[:30]:
            print(" ", i)
    else:
        print("=== 未发现异常 ===")

    sample = con.execute("SELECT ts_code FROM daily_bar LIMIT 1").fetchall()
    if sample:
        code = sample[0][0]
        gaps = check_gaps(con, code)
        print(f"\n{code} 日期缺口: {len(gaps)} 天" + ("（正常，停牌所致）" if len(gaps) else ""))
    con.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
