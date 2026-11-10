@echo off
REM Launcher for Windows Task Scheduler: runs the daily wildfire pipeline with the project's venv.
REM @echo off above hides each command as it runs, so the console stays clean.

REM Move into the project root: %~dp0 is this script's folder (scripts\), so ".." is the project.
cd /d "%~dp0.."

REM Run the pipeline with the venv's own Python (no activation needed).
".venv\Scripts\python.exe" -m src.pipeline.run_daily

REM Pass the pipeline's exit code (0 ok, 2 blocked, 1 failed) to Task Scheduler's "Last Run Result".
exit /b %ERRORLEVEL%