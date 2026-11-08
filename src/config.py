"""全局配置与路径。"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
DB_PATH = DATA_DIR / "stock.duckdb"

for _p in (DATA_DIR, REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# 首次全量回溯的年数
FULL_HISTORY_YEARS = 3
# 增量默认回看的交易日数（多拉几天做重叠去重，防止漏数据）
INCREMENT_LOOKBACK_DAYS = 30
# 东财接口并发数，过高会被限流
DEFAULT_WORKERS = 4
