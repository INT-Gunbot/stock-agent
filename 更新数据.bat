@echo off
chcp 65001 >nul 2>nul
cd /d "%~dp0"
title Stock Agent - Update Data

set PYTHONIOENCODING=utf-8
where python >nul 2>nul
if errorlevel 1 (
    echo   [ERROR] python not found in PATH.
    pause
    exit /b 1
)

echo.
echo   [1/5] Update quotes + daily report   (about 5-15 min)
echo   ------------------------------------------------------
python -m scripts.run_daily --update
if errorlevel 1 (
    echo.
    echo   [FAILED] update aborted. See the error above.
    pause
    exit /b 1
)

echo.
echo   [2/5] Rebuild the pick list   (about 1 min)
echo   ------------------------------------------------------
python -m scripts.run_select

echo.
echo   [3/5] Rebuild the portfolio checkup
echo   ------------------------------------------------------
python -m scripts.run_checkup

echo.
echo   [4/5] Fill announcement bodies + rebuild RAG index
echo         (today + yesterday, about 2-8 min. Close to zero
echo          if you already ran it after the close)
echo   ------------------------------------------------------
python -m scripts.fill_bodies

echo.
echo   [5/5] Update ETF daily bars   (about 4-6 min for 1664 funds)
echo         Runs after 15:00 so the close price is final.
echo   ------------------------------------------------------
python -m src.etf update

echo.
echo   Done. Reports are in the reports/ folder.
echo   Now run "启动工作台.bat" to view them.
echo.
pause
