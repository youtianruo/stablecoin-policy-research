@echo off
setlocal DisableDelayedExpansion

REM Stablecoin Policy Research - One-Click Runner
REM This script sets up the environment and runs the full pipeline

echo.
echo 🚀 Stablecoin Policy Research Pipeline
echo ======================================
echo.


REM --- Check Python ---
echo [STEP] Checking Python installation...
py --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Python not found. Please install Python 3.8+ and add it to PATH.
  goto END_ERR
)
echo [OK] Python found

REM --- Install dependencies directly (no virtual environment for simplicity) ---
echo [STEP] Installing dependencies...
echo [INFO] Installing core dependencies...
py -m pip install pandas>=2.0.0 numpy>=1.24.0 requests>=2.31.0 yfinance>=0.2.18 pyyaml>=6.0 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [WARN] Core dependencies installation had issues, trying individual packages...
  py -m pip install pandas >nul 2>&1
  py -m pip install numpy >nul 2>&1
  py -m pip install requests >nul 2>&1
  py -m pip install yfinance >nul 2>&1
  py -m pip install pyyaml >nul 2>&1
)

echo [INFO] Installing additional dependencies...
py -m pip install matplotlib seaborn plotly scipy scikit-learn statsmodels arch >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [WARN] Some additional dependencies may not be installed, but continuing...
)

echo [INFO] Installing project dependencies...
py -m pip install -e . >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [WARN] Project installation failed, trying requirements.txt...
  py -m pip install -r requirements.txt >nul 2>&1
)
echo [OK] Dependencies installation completed

REM --- Create .env if needed ---
if not exist ".env" (
  if exist ".env.example" (
    echo [INFO] Creating .env from .env.example...
    copy .env.example .env >nul
    echo [INFO] Please edit .env to add your API keys
  ) else (
    echo [INFO] No .env.example found; skipping .env creation
  )
)

REM --- Test multi-provider fetcher ---
echo [STEP] Testing multi-provider data fetcher ...
py test_fetcher_simple.py
if %ERRORLEVEL% NEQ 0 (
  echo [WARN] Multi-provider test had issues, but continuing...
)

REM --- Run pipeline steps ---
echo [STEP] Ingest market data ...
py -m src.pipelines.run_ingest --config configs\config.yaml || goto END_ERR

echo [STEP] Build features ...
py -m src.pipelines.run_features --config configs\config.yaml || goto END_ERR

echo [STEP] Run analysis ...
py -m src.pipelines.run_analysis --config configs\model_event_study.yaml || goto END_ERR

echo.
echo ✅ Done. Outputs should be in:
echo   data\raw\market_coingecko.parquet
echo   data\interim\market_features.parquet
echo   data\processed\analysis_results.parquet
echo.
echo 📊 Check notebooks\10_figures.ipynb for visualizations
echo.

goto END_OK

:END_ERR
echo.
echo ❌ Pipeline failed. Check the error messages above.
echo.
echo 🔧 Troubleshooting:
echo   1. Check your internet connection
echo   2. Verify API keys in .env file
echo   3. Run: py test_fetcher_simple.py
echo   4. Run: py -m src.data.fetch_coingecko
echo   5. Run: py check_deps.py
echo.
pause
exit /b 1

:END_OK
echo 🎉 Pipeline completed successfully!
pause
exit /b 0
