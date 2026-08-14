@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from python.org
  echo and enable "Add Python to PATH", then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment...
  py -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing or updating required packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo Starting MEA Analysis Workbench...
".venv\Scripts\python.exe" run_gui.py
exit /b %errorlevel%

:failed
echo.
echo Setup failed. Review the error above.
pause
exit /b 1
