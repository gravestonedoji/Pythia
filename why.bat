@echo off
REM Double-click to see Pythia's latest predictions and the reasoning behind each.
REM No API key needed (reads the local database only).
cd /d "%~dp0"
".venv\Scripts\pythia.exe" why %*
echo.
pause
