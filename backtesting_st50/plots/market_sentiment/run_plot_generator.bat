@echo off
echo Creating Enhanced NIFTY50 Plot with CPR...
echo.
cd /d "%~dp0"

echo Installing required packages...
..\..\..\venv\Scripts\python.exe -m pip install yfinance

echo.
echo Processing CSV data and generating plot...
..\..\..\venv\Scripts\python.exe process_csv_and_create_plot.py

if %errorlevel% equ 0 (
    echo.
    echo Opening the generated HTML file...
    start nifty50_plot.html
) else (
    echo.
    echo Error occurred while processing. Please check the error messages above.
)
echo.
pause
