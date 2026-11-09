@echo off
rem ============================================================
rem  nlpbase environment setup  (double-click to run)
rem
rem  Optional arguments, pass after the file name in a prompt:
rem     setup.bat -DryRun       show what would be done
rem     setup.bat -Cpu          install CPU build (no NVIDIA GPU)
rem     setup.bat -Index cu128  older driver, drop one CUDA level
rem ============================================================
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo   nlpbase environment setup
echo ============================================================
echo.

where powershell >nul 2>&1
if errorlevel 1 (
    echo [x] PowerShell not found.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set RC=%ERRORLEVEL%

echo.
if not "%RC%"=="0" (
    echo [x] Setup failed with exit code %RC%
    echo     Common causes:
    echo       conda not on PATH  - run from Anaconda Prompt
    echo       network blocked    - see condarc.example
    echo       GPU wheel mismatch - retry: setup.bat -Index cu128
) else (
    echo [ok] Done. Next: conda activate nlpbase
    echo      Then re-download model and dataset, see the guide.
)
echo.
pause
endlocal
