@echo off
rem ============================================================
rem  Run Qwen3-4B.py with the nlpbase interpreter.
rem  Double-click this file instead of clicking Run in the IDE,
rem  so the correct virtual environment is always used.
rem ============================================================
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "PY=D:\Anaconda3\envs\nlpbase\python.exe"
if not exist "%PY%" set "PY="

if defined PY (
    echo [run] %PY%
    "%PY%" "Qwen3-4B.py" %*
) else (
    echo [run] conda run -n nlpbase python
    conda run -n nlpbase python "Qwen3-4B.py" %*
)
echo.
pause
endlocal
