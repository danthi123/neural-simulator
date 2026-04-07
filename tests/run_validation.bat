@echo off
echo ============================================
echo  Neural Simulator GPU Validation Suite
echo ============================================
echo.

:: Try conda environment first, then system Python
if exist "%~dp0..\.conda\Scripts\python.exe" (
    set PYTHON=%~dp0..\.conda\Scripts\python.exe
    echo Using conda Python: %PYTHON%
) else (
    set PYTHON=python
    echo Using system Python
)

echo.
echo --- Quick Validation (fast, ~60 sec) ---
echo.
%PYTHON% "%~dp0validate_gpu.py" --quick

echo.
echo ============================================
echo  Results saved to: tests\validation_results.json
echo  Run without --quick for full validation
echo ============================================
echo.
pause
