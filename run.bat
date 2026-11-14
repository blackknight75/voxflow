@echo off
rem VoxFlow launcher (uses the project venv; pythonw keeps the console hidden - use run.bat --console to debug)
set "ROOT=%~dp0"
if /i "%~1"=="--console" (
  "%ROOT%.venv\Scripts\python.exe" -m voxflow %2 %3 %4 %5
) else (
  start "" "%ROOT%.venv\Scripts\pythonw.exe" -m voxflow %*
)
