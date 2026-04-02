@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [ERROR] Python venv not found at "%ROOT%.venv\Scripts\python.exe"
  echo Create it first: python -m venv .venv
  pause
  exit /b 1
)

if not exist "%ROOT%interface\gui.py" (
  echo [ERROR] GUI module not found at "%ROOT%interface\gui.py"
  pause
  exit /b 1
)

echo [INFO] Wait for the brain before starting the mouth.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(25); $ok=$false; while((Get-Date)-lt $deadline){ try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',11434); if($c.Connected){ $c.Close(); $ok=$true; break } } catch {} ; Start-Sleep -Milliseconds 500 }; if(-not $ok){ Write-Host '[WARN] Brain not reachable on 127.0.0.1:11434, starting GUI anyway.' }"

echo [INFO] Starting Sage v5 GUI...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$exists = @(Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like '*Sage v5 Desktop GUI*' }).Count -gt 0; if($exists){ exit 0 } else { exit 1 }"
if %errorlevel%==0 (
  echo [INFO] Sage v5 GUI already running. Skipping duplicate launch.
  endlocal
  exit /b 0
)
"%ROOT%.venv\Scripts\python.exe" -m interface.gui

endlocal
