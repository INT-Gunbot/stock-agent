"""为缺失复权因子的股票兜底填充。

背景：新浪 hfq-factor 不覆盖北交所（43/83/87/92 开头）与 CDR（689 开头），
全市场约 350 只拉不到。这些票如果因子为空，get_price 里 ffill/bfill 之后
adj_close 会变 NaN，一路传染到均线、技术面、日报，最终整份报告崩掉。

兜底策略（按优先级）：
1. 东财后复权价 ÷ 原始价 反推因子——精度不如新浪阶梯因子（东财序列有收益压缩），
   但至少是连续值，比 None 强得多
2. 东财也拉不到时，因子置 1.0（等于不做复权调整），并在报告里标注

这两种情况都会在质量检查里列为"因子来源降级"，不影响日报生成，
但相关个股的技术指标精度会下降，不应用它们做选股打分。
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.db import connect, exec_dml
from src.sources import akshare_source as src


def _fallback(ts_code: str, dates: list, closes: list) -> tuple[str, pd.DataFrame, str]:
    symbol = ts_code.split(".")[0]
    try:
        hfq = src._hist(symbol, "19900101", "20991231", "hfq")
        if hfq is not None and not hfq.empty and "close" in hfq.columns:
            h = hfq[["trade_date", "close"]].rename(columns={"close": "hfq_close"})
            h["hfq_close"] = pd.to_numeric(h["hfq_close"], errors="coerce")
            df = pd.DataFrame({"trade_date": dates, "close": closes})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            h["trade_date"] = pd.to_datetime(h["trade_date"]).dt.date
            m = df.merge(h, on="trade_date", how="left")
            ratio = (m["hfq_close"] / m["close"]).where(lambda x: x > 0)
            if ratio.notna().sum() > 0:
                out = pd.DataFrame({
                    "ts_code": ts_code,
                    "trade_date": m["trade_date"],
                    "adj_factor": ratio.ffill().bfill().fillna(1.0),
                })
                return ts_code, out, "em_hfq"
    except Exception:
        pass
    out = pd.DataFrame({
        "ts_code": ts_code,
        "trade_date": pd.to_datetime(pd.Series(dates)).dt.date,
        "adj_factor": 1.0,
    })
    return ts_code, out, "none"


def main() -> int:
    con = connect()
    missing = [r[0] for r in con.execute(
        """
        SELECT DISTINCT b.ts_code FROM daily_bar b
        LEFT JOIN adj_factor a ON a.ts_code = b.ts_code
        WHERE a.ts_code IS NULL ORDER BY 1
        """
    ).fetchall()]
    if not missing:
        print("没有缺失因子的股票")
        con.close()
        return 0

    print(f"缺失因子 {len(missing)} 只，开始兜底")
    payload = {}
    for tc in missing:
        rows = con.execute(
            "SELECT trade_date, close FROM daily_bar WHERE ts_code = ? ORDER BY 1", [tc]
        ).fetchall()
        if rows:
            payload[tc] = ([r[0] for r in rows], [r[1] for r in rows])

    frames, modes = [], {"em_hfq": 0, "none": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fallback, tc, d, c): tc for tc, (d, c) in payload.items()}
        for fut in as_completed(futures):
            done += 1
            try:
                _, df, mode = fut.result()
                frames.append(df)
                modes[mode] = modes.get(mode, 0) + 1
            except Exception as e:
                print(f"  {futures[fut]} 失败: {str(e)[:80]}")
            if done % 100 == 0:
                print(f"  进度 {done}/{len(payload)}")

    if frames:
        allf = pd.concat(frames, ignore_index=True)
        con.register("_fb", allf)
        try:
            exec_dml(con, "INSERT INTO adj_factor SELECT * FROM _fb")
        finally:
            con.unregister("_fb")
        print(f"写入 {len(allf)} 行：东财兜底 {modes.get('em_hfq',0)} 只，"
              f"无复权 {modes.get('none',0)} 只")

    left = con.execute(
        """
        SELECT count(DISTINCT b.ts_code) FROM daily_bar b
        LEFT JOIN adj_factor a ON a.ts_code = b.ts_code WHERE a.ts_code IS NULL
        """
    ).fetchall()[0][0]
    print(f"仍缺因子：{left} 只")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
