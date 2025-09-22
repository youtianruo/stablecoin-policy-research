@echo off
setlocal DisableDelayedExpansion

echo Testing batch script components...

echo [TEST] Python check...
py --version
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Python not found
  exit /b 1
)
echo [OK] Python found

echo [TEST] Dependencies check...
py check_deps.py
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Dependencies missing
  exit /b 1
)
echo [OK] Dependencies OK

echo [TEST] Multi-provider fetcher...
py test_fetcher_simple.py
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Multi-provider test failed
  exit /b 1
)
echo [OK] Multi-provider fetcher OK

echo [TEST] Ingest pipeline...
py -m src.pipelines.run_ingest --config configs\config.yaml
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Ingest pipeline failed
  exit /b 1
)
echo [OK] Ingest pipeline OK

echo.
echo 🎉 All tests passed! The batch script should work.
pause
