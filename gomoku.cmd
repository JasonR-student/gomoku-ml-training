@echo off
setlocal
set "PROJECT_DIR=%~dp0"
if "%1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%scripts\gomoku.ps1" download
  exit /b %ERRORLEVEL%
)
if /I "%1"=="config" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%scripts\gomoku.ps1" config
  exit /b %ERRORLEVEL%
)
if /I "%1"=="download" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%scripts\gomoku.ps1" download
  exit /b %ERRORLEVEL%
)
echo Usage: gomoku download
exit /b 2
