@echo off
rem Runs baixar_resultados.ps1 without changing the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0baixar_resultados.ps1" %*
pause
