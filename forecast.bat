@echo off
REM Double-click to run the daily forecast for the whole watchlist.
REM Needs ANTHROPIC_API_KEY in .env (this step calls the model).
cd /d "%~dp0"
".venv\Scripts\pythia.exe" forecast %*
echo.
pause
