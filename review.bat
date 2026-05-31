@echo off
REM Double-click to print the track record: leaderboard + full call log.
REM No API key needed (reads the local database only).
cd /d "%~dp0"
".venv\Scripts\pythia.exe" review %*
echo.
pause
