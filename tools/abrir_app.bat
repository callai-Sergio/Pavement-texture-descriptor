@echo off
rem Runs abrir_app.ps1 without changing the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0abrir_app.ps1" %*
pause
