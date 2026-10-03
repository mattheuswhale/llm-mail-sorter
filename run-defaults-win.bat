@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Virtual environment not found.
    echo Expected:
    echo   %PYTHON%
    echo.
    echo Create it first with:
    echo   python312 -m venv .venv
    pause
    exit /b 1
)

"%PYTHON%" run.py %*

endlocal