@echo off
echo Creating Strategy TradingView Visualization...
echo.

cd /d "%~dp0"

echo Processing strategy CSV data and generating plot...
..\..\..\venv\Scripts\python.exe run_tradeview_strategy.py

if %errorlevel% equ 0 (
    echo.
    echo Opening the generated HTML file...
    start NIFTY25O2025300CE_strategy.html
) else (
    echo.
    echo Error occurred while processing. Please check the error messages above.
)
echo.
pause

