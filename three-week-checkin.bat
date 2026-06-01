@echo off
REM One-time 3-week check-in reminder (run by a Windows scheduled task).
REM Dumps the current scoreboard to a file on the Desktop and opens it.
cd /d "%~dp0"
set "OUT=%USERPROFILE%\Desktop\Pythia 3-week check-in.txt"
> "%OUT%" echo PYTHIA - 3-WEEK CHECK-IN
>> "%OUT%" echo Generated %DATE% %TIME%
>> "%OUT%" echo.
>> "%OUT%" echo It has been about three weeks since Pythia started forecasting.
>> "%OUT%" echo Time to decide whether to build the self-review loop (version 1).
>> "%OUT%" echo.
>> "%OUT%" echo WHAT TO DO:
>> "%OUT%" echo   1. Read the scoreboard below. Is Pythia beating the baselines?
>> "%OUT%" echo      (lower "avg Brier" is better; 0.25 = a coin flip)
>> "%OUT%" echo   2. Double-click why.bat to re-read its recent reasoning.
>> "%OUT%" echo   3. Open Claude Code in the Pythia folder and say you want to
>> "%OUT%" echo      start version 1 (the weekly self-review). The full plan is saved.
>> "%OUT%" echo.
>> "%OUT%" echo ============================================================
>> "%OUT%" echo SCOREBOARD (pythia review)
>> "%OUT%" echo ============================================================
".venv\Scripts\pythia.exe" review --no-log >> "%OUT%" 2>&1
start "" notepad "%OUT%"
