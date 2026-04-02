@echo off
setlocal

set "TASK_NAME=Sage v5 Autostart"
set "MASTER_BAT=D:\Sage_v5\Sage_v5_Master.bat"

if not exist "%MASTER_BAT%" (
  echo [ERROR] Master launcher not found: "%MASTER_BAT%"
  exit /b 1
)

echo [INFO] Removing existing task (if any)...
schtasks /Query /TN "%TASK_NAME%" >nul 2>&1
if %errorlevel%==0 (
  schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1
)

echo [INFO] Creating task with 60s logon delay (interactive GUI)...
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "%MASTER_BAT%" ^
  /SC ONLOGON ^
  /DELAY 0001:00 ^
  /RL HIGHEST ^
  /IT ^
  /F

if errorlevel 1 (
  echo [ERROR] Failed to create task. Run this script as Administrator.
  exit /b 1
)

echo [OK] Task created.
echo [INFO] Task details:
schtasks /Query /TN "%TASK_NAME%" /V /FO LIST

echo [INFO] Launching task now for quick test...
schtasks /Run /TN "%TASK_NAME%"

endlocal
exit /b 0
