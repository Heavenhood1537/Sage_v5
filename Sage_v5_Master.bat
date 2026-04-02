@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "ENGINE_BAT=%ROOT%Start-Sage_v5-Engine.bat"
set "GUI_BAT=%ROOT%Start-Sage_v5-GUI.bat"
set "PORT=11434"
set "BOOT_DELAY_SECS=30"
set "INTERNET_WAIT_SECS=35"
set "WIFI_PROFILE=TELUS7480"
set "WIFI_IFACE=Wi-Fi 2"

if not exist "%ENGINE_BAT%" (
  echo [WARN] Engine launcher not found: "%ENGINE_BAT%"
  echo [WARN] Continuing with GUI-only startup.
)

if not exist "%GUI_BAT%" (
  echo [ERROR] GUI launcher not found: "%GUI_BAT%"
  exit /b 1
)

if /I not "%~1"=="--now" (
  echo [INFO] Boot delay: waiting %BOOT_DELAY_SECS%s for Wi-Fi/network readiness...
  timeout /t %BOOT_DELAY_SECS% /nobreak >nul
)

echo [0/3] Checking Wi-Fi link...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$raw = netsh wlan show interfaces | Out-String; if($raw -match 'State\s*:\s*connected'){ exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo [INFO] Wi-Fi not connected yet. Nudging auto-connect to "%WIFI_PROFILE%" on "%WIFI_IFACE%"...
  netsh wlan connect name="%WIFI_PROFILE%" interface="%WIFI_IFACE%" >nul 2>&1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(25); $ok=$false; while((Get-Date)-lt $deadline){ $raw = netsh wlan show interfaces | Out-String; if($raw -match 'State\s*:\s*connected'){ $ok=$true; break }; Start-Sleep -Milliseconds 800 }; if($ok){ exit 0 } else { exit 1 }"
  if errorlevel 1 (
    echo [WARN] Wi-Fi still not connected after nudge. Continuing startup anyway.
  ) else (
    echo [INFO] Wi-Fi connected.
  )
) else (
  echo [INFO] Wi-Fi already connected.
)

echo [0.5/3] Waiting for internet readiness (DNS/route)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(%INTERNET_WAIT_SECS%); $ok=$false; while((Get-Date)-lt $deadline){ try { $null=[System.Net.Dns]::GetHostEntry('www.msftconnecttest.com'); $c = New-Object Net.Sockets.TcpClient; $c.Connect('1.1.1.1',443); if($c.Connected){ $c.Close(); $ok=$true; break } } catch {}; Start-Sleep -Milliseconds 1200 }; if($ok){ exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo [WARN] Internet still not ready after %INTERNET_WAIT_SECS%s. Continuing startup anyway.
) else (
  echo [INFO] Internet check passed.
)

echo [1/3] Checking Ollama service on 127.0.0.1:%PORT%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',%PORT%); if($c.Connected){ $c.Close(); $ok=$true } } catch {}; if($ok){ exit 0 } else { exit 1 }"
if errorlevel 1 (
  if exist "%ENGINE_BAT%" (
    echo [INFO] Ollama not detected. Starting helper launcher...
    start "Sage v5 Ollama" /min /belownormal "%ENGINE_BAT%"
  ) else (
    echo [WARN] Helper launcher missing; skipping service start.
  )
) else (
  echo [INFO] Ollama service already running.
)

echo [2/3] Waiting for Ollama readiness...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(45); $ok=$false; while((Get-Date)-lt $deadline){ try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',%PORT%); if($c.Connected){ $c.Close(); $ok=$true; break } } catch {} ; Start-Sleep -Milliseconds 500 }; if($ok){ exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo [WARN] Ollama did not become reachable in time. Continuing to GUI.
)

echo [3/3] Launching Sage v5 GUI...
start "Sage v5 GUI" "%GUI_BAT%"

echo [DONE] Master launcher started.
exit /b 0
