@echo off
echo Starting Strategy Plotter...
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Activate the virtual environment
call ..\venv\Scripts\activate.bat

REM Run the Python strategy plotter script using the virtual environment Python
..\venv\Scripts\python.exe strategy_plotter.py

echo.
echo Strategy plotting process finished.
pause
