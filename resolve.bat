@echo off
REM Double-click to settle every matured forecast against the real close.
REM No API key needed (uses prices + the local database only).
cd /d "%~dp0"
".venv\Scripts\pythia.exe" resolve %*
echo.
pause
