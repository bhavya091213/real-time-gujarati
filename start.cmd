@echo off
REM guju-sub launcher for Windows 10/11. Installs on first run, then starts the server.
REM
REM   start.cmd [script flags] [server flags]
REM
REM Script flags:
REM   --setup             run install-windows.cmd first even if .venv exists (idempotent)
REM   --skip-models       passed to the installer when it runs
REM   --open              open the mic page and /display in the browser once ready
REM   --display-only      open only /display once ready
REM   --threads N         set GUJUSUB_THREADS=N
REM   --model-dir PATH    set GUJUSUB_MODEL_DIR=PATH
REM   --replay FILE ...   run tools\replay.py FILE ... instead of the server
REM   --verify            load both models once (tools\prefetch_models.py --verify) and exit
REM   --keep-port         do not stop an existing listener on the server port first
REM   --test              run pytest and exit
REM   -h, --help          this text plus the live server flag list
REM
REM Before starting the server, anything already listening on the port (--port,
REM default 8765) is stopped, unless --keep-port is given.
REM
REM Any other flag (--device, --engines, --port, --no-translate, --no-filter, ...)
REM is passed to the server unchanged.

setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set SETUP=0
set SKIP_MODELS=0
set OPEN_MAIN=0
set OPEN_DISPLAY=0
set MODE=server
set KEEP_PORT=0
set PORT=8765
set "SERVER_ARGS="
set "REPLAY_ARGS="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="-h"            goto help
if /i "%~1"=="--help"        goto help
if /i "%~1"=="--setup"       (set SETUP=1& shift& goto parse)
if /i "%~1"=="--skip-models" (set SKIP_MODELS=1& shift& goto parse)
if /i "%~1"=="--open"        (set OPEN_MAIN=1& set OPEN_DISPLAY=1& shift& goto parse)
if /i "%~1"=="--display-only" (set OPEN_DISPLAY=1& shift& goto parse)
if /i "%~1"=="--keep-port"  (set KEEP_PORT=1& shift& goto parse)
if /i "%~1"=="--verify"      (set MODE=verify& shift& goto parse)
if /i "%~1"=="--test"        (set MODE=test& shift& goto parse)
if /i "%~1"=="--threads" (
  if "%~2"=="" (echo error: --threads needs a value& exit /b 2)
  set "GUJUSUB_THREADS=%~2"& shift& shift& goto parse
)
if /i "%~1"=="--model-dir" (
  if "%~2"=="" (echo error: --model-dir needs a value& exit /b 2)
  set "GUJUSUB_MODEL_DIR=%~2"& shift& shift& goto parse
)
if /i "%~1"=="--replay" (
  if "%~2"=="" (echo error: --replay needs an audio file& exit /b 2)
  set MODE=replay& shift& goto replayargs
)
if /i "%~1"=="--port" (
  if "%~2"=="" (echo error: --port needs a value& exit /b 2)
  set "PORT=%~2"
  set SERVER_ARGS=%SERVER_ARGS% --port %~2
  shift& shift& goto parse
)
set SERVER_ARGS=%SERVER_ARGS% %1
shift
goto parse

:replayargs
REM everything after --replay is forwarded to tools\replay.py
if "%~1"=="" goto parsed
set REPLAY_ARGS=%REPLAY_ARGS% %1
shift
goto replayargs

:parsed

REM ---- make sure the venv exists (install on first run) -----------------------
set NEED_INSTALL=0
if "%SETUP%"=="1" set NEED_INSTALL=1
if not exist "%PY%" set NEED_INSTALL=1
if "%NEED_INSTALL%"=="1" (
  echo ==^> running installer
  if "%SKIP_MODELS%"=="1" (
    call install-windows.cmd --skip-models || exit /b 1
  ) else (
    call install-windows.cmd || exit /b 1
  )
)
if not exist "%PY%" (
  echo error: %PY% missing after install
  exit /b 1
)

if "%MODE%"=="verify" (
  "%PY%" tools\prefetch_models.py --verify
  exit /b %errorlevel%
)
if "%MODE%"=="test" (
  "%PY%" -m pytest
  exit /b %errorlevel%
)
if "%MODE%"=="replay" (
  "%PY%" tools\replay.py%REPLAY_ARGS%
  exit /b %errorlevel%
)

REM ---- server -----------------------------------------------------------------
if "%KEEP_PORT%"=="0" call :freeport || exit /b 1
if "%OPEN_MAIN%"=="0" if "%OPEN_DISPLAY%"=="0" goto run

REM Detached helper: poll the mic page until it answers 200 (90 s max), then open
REM the browser. The server itself runs in the foreground below so Ctrl-C stops it.
set "OPEN_PS=$u='http://localhost:%PORT%/'; $ok=$false; for($i=0;$i -lt 90 -and -not $ok;$i++){ try{ $ok=((Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2).StatusCode -eq 200) }catch{ Start-Sleep -Seconds 1 } }; if($ok){ if(%OPEN_MAIN% -eq 1){ Start-Process $u }; if(%OPEN_DISPLAY% -eq 1){ Start-Process ($u + 'display') } } else { Write-Host 'server not ready after 90 s; browser not opened' }"
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "%OPEN_PS%"

:run
"%PY%" -m gujusub.server%SERVER_ARGS%
exit /b %errorlevel%

:freeport
set "KILLED="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do call :killpid %%p
if not defined KILLED exit /b 0
timeout /t 1 >nul
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if errorlevel 1 exit /b 0
timeout /t 2 >nul
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if errorlevel 1 exit /b 0
echo error: port %PORT% is still in use after stopping PID(s)%KILLED%; stop it or pass --keep-port
exit /b 1

:killpid
REM %~1 = PID from netstat (the same PID can appear on several lines, e.g. IPv4 + IPv6)
if "%~1"=="0" exit /b 0
echo  %KILLED% | findstr /C:" %~1 " >nul && exit /b 0
set "KILLED=%KILLED% %~1"
taskkill /PID %~1 /F >nul 2>&1 && echo ==^> stopped PID %~1 listening on port %PORT%
exit /b 0

:help
call :usage
if exist "%PY%" (
  echo.
  echo Server flags ^(python -m gujusub.server --help^):
  "%PY%" -m gujusub.server --help
) else (
  echo.
  echo ^(server flags are listed once .venv exists; run start.cmd --setup^)
)
exit /b 0

:usage
echo start.cmd [script flags] [server flags]
echo.
echo Script flags:
echo   --setup             run install-windows.cmd first even if .venv exists
echo   --skip-models       passed to the installer when it runs
echo   --open              open the mic page and /display once the server is ready
echo   --display-only      open only /display
echo   --threads N         set GUJUSUB_THREADS=N
echo   --model-dir PATH    set GUJUSUB_MODEL_DIR=PATH
echo   --replay FILE ...   run tools\replay.py FILE ... instead of the server
echo   --verify            load both models once and exit
echo   --keep-port         do not stop an existing listener on the server port first
echo   --test              run pytest and exit
echo   -h, --help          this text plus the live server flag list
echo.
echo Anything already listening on the server port (--port, default 8765) is
echo stopped first unless --keep-port is given.
echo.
echo Any other flag (--device, --engines, --port, --no-translate, --no-filter, ...)
echo is passed to the server unchanged.
exit /b 0
