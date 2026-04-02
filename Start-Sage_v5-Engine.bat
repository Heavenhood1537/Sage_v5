@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PORT=11434"
set "MODEL=qwen2.5:3b"
set "ONLINE=0"

echo [INFO] Checking Ollama service on 127.0.0.1:%PORT% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',%PORT%); if($c.Connected){ $c.Close(); $ok=$true } } catch {}; if($ok){ exit 0 } else { exit 1 }"
if not errorlevel 1 (
	echo [INFO] Ollama is already running.
	set "ONLINE=1"
)

where ollama >nul 2>&1
if errorlevel 1 (
	echo [ERROR] Ollama CLI not found in PATH.
	echo [ERROR] Install Ollama or add it to PATH, then retry.
	exit /b 1
)

if "%ONLINE%"=="0" (
	echo [INFO] Starting Ollama service helper...
	start "" /min ollama serve

	echo [INFO] Waiting for Ollama readiness...
	powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(20); $ok=$false; while((Get-Date)-lt $deadline){ try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',%PORT%); if($c.Connected){ $c.Close(); $ok=$true; break } } catch {}; Start-Sleep -Milliseconds 500 }; if($ok){ exit 0 } else { exit 1 }"
	if errorlevel 1 (
		echo [WARN] Ollama was started but is not reachable yet.
		exit /b 1
	)
)

echo [INFO] Pulling required model: %MODEL%
ollama pull %MODEL%
if errorlevel 1 (
	echo [ERROR] Failed to pull %MODEL%
	exit /b 1
)

echo [INFO] Warming model: %MODEL%
ollama run %MODEL% "Respond with exactly: ready" >nul

echo [INFO] Ollama is online.
exit /b 0
