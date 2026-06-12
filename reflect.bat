@echo off
REM Double-click to run the weekly self-review: Opus reads the graded record
REM and distills lessons into lessons.txt, turning the coached arm on.
REM Needs ANTHROPIC_API_KEY in .env (this step calls the model).
REM Tip: run "reflect.bat --dry-run" to preview the lessons without saving.
cd /d "%~dp0"
".venv\Scripts\pythia.exe" reflect %*
echo.
pause
