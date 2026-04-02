@echo off
setlocal

set "TASK_NAME=Sage v5 Autostart"

schtasks /Query /TN "%TASK_NAME%" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Task "%TASK_NAME%" not found.
  exit /b 0
)

schtasks /Delete /TN "%TASK_NAME%" /F
if errorlevel 1 (
  echo [ERROR] Failed to delete task. Try Run as Administrator.
  exit /b 1
)

echo [OK] Task removed.
endlocal
exit /b 0
