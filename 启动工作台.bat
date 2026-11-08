@echo off
chcp 65001 >nul 2>nul
cd /d "%~dp0"
title 炒股 Agent · 本地工作台
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo   ============================================================
echo    炒股 Agent · 本地工作台
echo   ============================================================
echo.

rem ============================================================
rem  1. Pick a Python that really imports the deps
rem
rem  Two lessons are baked into the order below.
rem
rem  a) `where python` is not enough. This machine carries the
rem     Microsoft Store stub (it opens the Store instead of
rem     running), conda's python (pandas yes, duckdb no) and a
rem     bundled python that has all three. PATH order is not
rem     something this script controls.
rem
rem  b) A python visible from a shell may be invisible on a real
rem     double-click, because the shell injects its own PATH.
rem     Guessing from PATH alone produced exactly that false pass,
rem     so absolute locations are probed too.
rem
rem  Every candidate is verified by importing. Never by version
rem  string, never by path guessing. First one that passes wins.
rem
rem  NOTE: comments stay ASCII on purpose. cmd mis-parses non-ASCII
rem  text inside rem blocks under chcp 65001 (it splits the line and
rem  runs the tail as a command). Chinese belongs in echo.
rem ============================================================
set "PY="
set "LOG=%TEMP%\stockagent_py_probe.txt"
>"%LOG%" echo   试过这些 Python：

rem --- 1a. project venv: fastest and fully pinned, always first
if exist ".venv\Scripts\python.exe" (
    set C=".venv\Scripts\python.exe"
    call :try
)
if defined PY goto :gotpy

rem --- 1b. every python on PATH, minus the Store stub
for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PY (
        if /i not "%%~dpP"=="%LOCALAPPDATA%\Microsoft\WindowsApps\" (
            set C="%%~fP"
            call :try
        )
    )
)
if defined PY goto :gotpy

rem --- 1c. the py launcher (picks the newest installed CPython)
where py >nul 2>nul
if not errorlevel 1 (
    set C=py -3
    call :try
)
if defined PY goto :gotpy

rem --- 1d. bundled python shipped with the WorkBuddy app
for /d %%D in ("%USERPROFILE%\.workbuddy\binaries\python\versions\*") do (
    if not defined PY if exist "%%~fD\python.exe" (
        set C="%%~fD\python.exe"
        call :try
    )
)
if defined PY goto :gotpy

rem --- 1e. conda base envs
for /d %%R in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if not defined PY if exist "%%~fR\python.exe" (
        set C="%%~fR\python.exe"
        call :try
    )
)
if defined PY goto :gotpy

rem --- 1f. conda named envs
for /d %%D in ("%USERPROFILE%\miniconda3\envs\*" "%USERPROFILE%\anaconda3\envs\*") do (
    if not defined PY if exist "%%~fD\python.exe" (
        set C="%%~fD\python.exe"
        call :try
    )
)
if defined PY goto :gotpy

rem --- 1g. ordinary installers
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*" "%ProgramFiles%\Python3*" "C:\Python3*") do (
    if not defined PY if exist "%%~fD\python.exe" (
        set C="%%~fD\python.exe"
        call :try
    )
)
if defined PY goto :gotpy

goto :nopy

rem ------------------------------------------------------------
rem  :try  C holds a ready-to-run command string.
rem  Pass = imports fine AND 3.10+. Anything else gets logged so
rem  the failure branch can say which one was found and why it
rem  was rejected, instead of a bare "not found".
rem ------------------------------------------------------------
:try
%C% -c "import sys, pandas, duckdb, akshare; raise SystemExit(0 if sys.version_info >= (3, 10) else 2)" >nul 2>nul
if not errorlevel 1 (
    set "PY=%C%"
    exit /b
)
>>"%LOG%" echo   - %C%
rem  No % formatting here on purpose: in a .bat a lone %x is read as a
rem  variable/for reference and gets swallowed (this broke once already,
rem  python then died with "unterminated string literal").
%C% -c "import sys, importlib.util as u; m=[n for n in ('pandas','duckdb','akshare') if u.find_spec(n) is None]; print('      Python', '.'.join(map(str, sys.version_info[:2])), '| missing:', ', '.join(m) if m else 'none - version below 3.10')" >>"%LOG%" 2>&1
exit /b

rem ============================================================
rem  2. Port 8765
rem
rem  Three cases, never kill blindly:
rem   - our own python is listening -> the workbench is already up;
rem     just open the page. Killing it would drop a live session.
rem   - something else holds it     -> report, let the user decide.
rem   - arg "restart"               -> then actually kill it.
rem ============================================================
:gotpy
if /i "%~1"=="check" (
    echo   [check] OK
    echo   Python : %PY%
    del "%LOG%" >nul 2>nul
    exit /b 0
)

set "OLDPID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /C:":8765 " ^| findstr /I "LISTENING"') do set "OLDPID=%%P"

if /i "%~1"=="restart" goto :killold
if not defined OLDPID goto :run

tasklist /FI "PID eq %OLDPID%" /NH 2>nul | findstr /I "python" >nul
if errorlevel 1 goto :busy

echo   工作台已经在运行，直接打开页面。
start "" http://127.0.0.1:8765/
echo.
echo   要重启（比如刚改过代码）：先关掉那个黑窗口，或双击时加 restart 参数。
ping -n 4 127.0.0.1 >nul
del "%LOG%" >nul 2>nul
exit /b 0

:killold
if not defined OLDPID goto :run
echo   停掉旧的进程（PID %OLDPID%）...
taskkill /PID %OLDPID% /F >nul 2>nul
ping -n 2 127.0.0.1 >nul
goto :run

:busy
echo   [ERROR] 端口 8765 被别的程序占用了（PID %OLDPID%）：
tasklist /FI "PID eq %OLDPID%" /NH
echo.
echo   先关掉它，或者改 web\app.py 里的 PORT 常量。
del "%LOG%" >nul 2>nul
pause
exit /b 1

rem ============================================================
rem  3. Start
rem ============================================================
:run
del "%LOG%" >nul 2>nul
if not defined DEEPSEEK_API_KEY if not defined ANTHROPIC_API_KEY (
    echo   [提示] 没检测到模型 key（DEEPSEEK_API_KEY / ANTHROPIC_API_KEY）。
    echo          将以「本地规则」模式启动：能搬数据、能排版，但不会推理。
    echo.
)

echo   Python : %PY%
echo   地址   : http://127.0.0.1:8765
echo   停止   : 关掉本窗口，或按 Ctrl+C
echo.
echo   正在启动...

%PY% -m web.app
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" (
    echo   [注意] 服务退出，代码 %CODE%。被强制结束的可以忽略，否则看上面的报错。
) else (
    echo   服务已停止。
)
echo   数据不是今天的，先跑一次「更新数据.bat」。
echo.
pause
exit /b %CODE%

rem ============================================================
rem  Failure branch
rem ============================================================
:nopy
echo   [ERROR] 没找到一个能用的 Python。
echo.
echo   要求：Python 3.10 以上，且装好 pandas / duckdb / akshare。
echo.
type "%LOG%"
del "%LOG%" >nul 2>nul
echo.
echo   两种解决办法（任选其一）：
echo.
echo     1) 给上面某个 Python 补装依赖，比如：
echo          "C:\路径\python.exe" -m pip install -r requirements.txt
echo.
echo     2) 建一个本项目专用环境（推荐，不污染系统 Python）：
echo          python -m venv .venv
echo          .venv\Scripts\python.exe -m pip install -r requirements.txt
echo        装好后再双击本脚本，它会优先用 .venv。
echo.
pause
exit /b 1
