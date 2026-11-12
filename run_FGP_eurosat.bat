@echo off
setlocal enabledelayedexpansion

:: ======================================================================
::  FGP x EuroSAT-LT-400 (exp) - seeds 1..30
::  Output: output\FGP\{log,pkl,csv}\EuroSAT-LT-400\
::  Place this file in the project root (same dir as FGP).
::  EDIT PYTHON_EXE below if your env path differs.
:: ======================================================================
:: use the activated conda env if available, otherwise fall back to the default interpreter
set "PYTHON_EXE=E:\Code\py311\python.exe"
if defined CONDA_PREFIX set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"

set "METHOD=FGP"
set "DATASET=EuroSAT"
set "LABEL=EuroSAT-LT-400"
set "LTMODE=exp"
set "SEEDS=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"

echo [Info] PYTHON_EXE=%PYTHON_EXE%
if not exist "%PYTHON_EXE%" (
    echo [Fatal] PYTHON_EXE not found: "%PYTHON_EXE%"
    echo         Please edit the PYTHON_EXE line at the top of this script.
    exit /b 2
)
if not exist "%~dp0%METHOD%\" (
    echo [Fatal] method folder not found: "%~dp0%METHOD%"
    exit /b 2
)

pushd "%~dp0%METHOD%"
call :run_task
set "RC=!errorlevel!"
popd
if "!RC!"=="130" goto :abort

echo.
echo [ALL DONE - %METHOD% x %LABEL%]
exit /b 0

:abort
echo.
echo [ABORT] Ctrl+C detected, all runs stopped.
exit /b 130

:: ======================================================================
::  run_task
::  Runs the seeds listed in SEEDS inside the current method folder.
:: ======================================================================
:run_task
set "OUTBASE=%~dp0output\%METHOD%\"

mkdir "%OUTBASE%log\%LABEL%" 2>nul
mkdir "%OUTBASE%pkl\%LABEL%" 2>nul
mkdir "%OUTBASE%csv\%LABEL%" 2>nul

for %%i in (%SEEDS%) do (
    set "LOGFILE=%OUTBASE%log\%LABEL%\%DATASET%_%%i.log"
    set "PKLFILE=%OUTBASE%pkl\%LABEL%\%DATASET%_%%i.pkl"
    set "CSVFILE=%OUTBASE%csv\%LABEL%\%DATASET%_%%i.csv"
    set "DONEFILE=%OUTBASE%pkl\%LABEL%\%DATASET%_%%i.done"

    if exist "!DONEFILE!" (
        echo [Skip] %DATASET% seed %%i already completed.
    ) else (
        echo [Start/Resume] %DATASET% seed %%i ...
        "%PYTHON_EXE%" monitor_subprocess.py --save "!CSVFILE!" --include-children "%PYTHON_EXE%" main.py --log_file "!LOGFILE!" --pkl_file "!PKLFILE!" --dataset %DATASET% --randomSeeds %%i --lt_mode %LTMODE% --disable_imb_test --population 100 --generation 50 --num_processes 10

        if !errorlevel! equ 130 (
            echo [ABORT] Ctrl+C detected, stopping remaining runs.
            exit /b 130
        )
        if !errorlevel! equ 0 (
            echo [Success] %DATASET% seed %%i finished.
            echo. > "!DONEFILE!"
        ) else (
            echo [Error/Interrupted] %DATASET% seed %%i not finished. Will resume next time.
        )
    )
)
exit /b 0
