@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
call "%ROOT%Sage_v5_Master.bat" %*
endlocal
