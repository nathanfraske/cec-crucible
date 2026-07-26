@echo off
REM Double-click entry point: runs the per-user installer with an execution
REM policy bypass so it works on a default Windows box without any prep.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Crucible.ps1" %*
echo.
pause
