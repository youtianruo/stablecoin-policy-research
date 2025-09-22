@echo off
REM ===== Stablecoin–Policy Research: Windows Runner (safe, no paren blocks) =====
setlocal DisableDelayedExpansion

echo.
echo === Stablecoin-Policy Research: Windows Runner ===
echo Current folder: %cd%
echo.

REM --- Sanity checks (files) ---
if not exist "configs\config.yaml" goto NO_CFG
if not exist "src\pipelines\run_ingest.py" goto NO_PIPE

REM --- Pick Python ---
where python >nul 2>nul
if %ERRORLEVEL%==0 (
  set PY=python
) else (
  where py >nul 2>nul
  if %ERRORLEVEL%==0 (
    set PY=py -3
  ) else (
    echo [ERROR] Python not found. Install from https://www.python.org/ (check "Add to PATH").
    goto END_ERR
  )
)
echo Using interpreter: %PY%

REM --- Create venv if missing ---
if exist ".\.venv\Scripts\activate.bat" goto ACTIVATE
echo [INFO] Creating virtual environment .venv ...
%PY% -m venv .venv
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Failed to create virtual environment.
  goto END_ERR
)

:ACTIVATE
call .\.venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Failed to activate .venv. Use Command Prompt (cmd.exe), not PowerShell.
  goto END_ERR
)
echo [INFO] venv activated.

REM --- Upgrade pip & install deps ---
echo [INFO] Upgrading pip ...
python -m pip install --upgrade pip || goto END_ERR

if exist requirements.txt (
  echo [INFO] Installing from requirements.txt ...
  pip install -r requirements.txt || goto END_ERR
) else (
  echo [INFO] Installing project (editable) ...
  pip install -e . || goto END_ERR
)

REM --- Prepare .env if missing ---
if not exist ".env" (
  if exist ".env.example" (
    copy /Y .env.example .env >nul
    echo [INFO] Created .env from .env.example
  ) else (
    echo [INFO] No .env.example found; skipping .env creation
  )
)

REM --- Run pipeline steps ---
echo [STEP] Ingest market data ...
python -m src.pipelines.run_ingest --config configs\config.yaml || goto END_ERR

echo [STEP] Build features ...
python -m src.pipelines.run_features --config configs\config.yaml || goto END_ERR

echo [STEP] Run analysis ...
python -m src.pipelines.run_analysis --config configs\model_event_study.yaml || goto END_ERR

echo.
echo ✅ Done. Outputs should be in:
echo   data\raw\market_coingecko.parquet
echo   data\interim\market_features.parquet
echo   data\processed\event_study_car.parquet
echo.
goto END_OK

:NO_CFG
echo [ERROR] Missing configs\config.yaml . Are you in the repo root?
goto END_ERR

:NO_PIPE
echo [ERROR] Missing src\pipelines\run_ingest.py . Are you in the repo root?
goto END_ERR

:END_OK
pause
endlocal
exit /b 0

:END_ERR
echo.
echo ❌ Failed. Review messages above. If running in PowerShell, try cmd.exe or run:
echo   .\run_all_windows.bat
pause
endlocal
exit /b 1