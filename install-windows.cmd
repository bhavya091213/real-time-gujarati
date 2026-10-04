@echo off
REM guju-sub installer for Windows 10/11.
REM
REM   install-windows.cmd                # venv + runtime deps + pre-download models
REM   install-windows.cmd --dev          # also install pytest/httpx/ruff and run the test suite
REM   install-windows.cmd --skip-models
REM
REM Safe to re-run: reuses an existing .venv and only installs what is missing.
REM
REM Prerequisites:
REM   * Python 3.11+ from https://www.python.org/downloads/windows/
REM     (tick "Add python.exe to PATH" and keep the "py launcher" option).
REM   * Microsoft C++ Build Tools (only "Desktop development with C++"):
REM     https://visualstudio.microsoft.com/visual-cpp-build-tools/
REM     IndicTransToolkit ships no Windows wheel, so pip compiles it from source.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set DEV=0
set SKIP_MODELS=0
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--dev"         set DEV=1
if /i "%~1"=="--skip-models" set SKIP_MODELS=1
if /i "%~1"=="-h"            goto help
if /i "%~1"=="--help"        goto help
shift
goto parse
:parsed

set MIN_MINOR=11

REM ---- 1. find a suitable Python ------------------------------------------
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
  for /f "delims=" %%v in ('".venv\Scripts\python.exe" --version') do echo ==^> reusing existing .venv ^(%%v^)
  goto deps
)

set "SYS_PY="
for %%v in (3.13 3.12 3.11) do (
  if not defined SYS_PY (
    py -%%v -c "import sys" >nul 2>&1 && set "SYS_PY=py -%%v"
  )
)
if not defined SYS_PY (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, %MIN_MINOR%) else 1)" >nul 2>&1 && set "SYS_PY=python"
)
if not defined SYS_PY (
  echo error: Python 3.%MIN_MINOR%+ not found. Install it from https://www.python.org/downloads/windows/
  echo        ^(tick "Add python.exe to PATH"^) and re-run this script.
  exit /b 1
)

for /f "delims=" %%v in ('%SYS_PY% --version') do echo ==^> creating .venv with %SYS_PY% ^(%%v^)
%SYS_PY% -m venv .venv || (echo error: could not create .venv & exit /b 1)
set "PY=.venv\Scripts\python.exe"

REM ---- 2. install dependencies --------------------------------------------
:deps
echo ==^> upgrading pip
"%PY%" -m pip install --quiet --upgrade pip || (echo error: pip upgrade failed & exit /b 1)

if "%DEV%"=="1" (
  echo ==^> installing runtime + dev dependencies ^(requirements-dev.txt^)
  "%PY%" -m pip install -r requirements-dev.txt
) else (
  echo ==^> installing runtime dependencies ^(requirements.txt^)
  "%PY%" -m pip install -r requirements.txt
)
if errorlevel 1 (
  echo.
  echo error: dependency install failed.
  echo.
  echo If the failure is in IndicTransToolkit ^("Microsoft Visual C++ 14.0 or greater
  echo is required"^), install the C++ Build Tools and re-run:
  echo     https://visualstudio.microsoft.com/visual-cpp-build-tools/
  echo     ^(select the "Desktop development with C++" workload^)
  exit /b 1
)

echo ==^> verifying imports
"%PY%" -c "from gujusub.translator import Translator; from gujusub.asr_engine import ASREngine" || (
  echo error: import check failed; see the traceback above
  exit /b 1
)

REM ---- 3. pre-download models (~1 GB; hub cache + real-file copy in %%USERPROFILE%%\.cache\gujusub)
if "%SKIP_MODELS%"=="0" (
  echo ==^> downloading models ^(skip with --skip-models^)
  "%PY%" tools\prefetch_models.py || (
    echo error: model download failed; check your network and re-run
    exit /b 1
  )
  echo ==^> loading models once to verify they work ^(ASR + translator, ~30 s^)
  "%PY%" tools\prefetch_models.py --verify || (
    echo error: model load check failed; see the traceback above
    exit /b 1
  )
)

REM ---- 4. optional: run the test suite --------------------------------------
if "%DEV%"=="1" (
  echo ==^> running tests
  "%PY%" -m pytest || exit /b 1
)

echo.
echo Done.
echo.
echo Next: start.cmd --open      ^(starts the server and opens the mic + display pages^)
echo.
echo Windows runs the models on CPU ^(--device mps is macOS only^).
exit /b 0

:help
echo install-windows.cmd [--dev] [--skip-models]
echo   --dev           also install pytest/httpx/ruff and run the test suite
echo   --skip-models   do not pre-download the ~1 GB of models
exit /b 0
