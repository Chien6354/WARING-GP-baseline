@echo off
setlocal enabledelayedexpansion

:: ======================================================================
::  dtgpn x MNIST-LT-200 (fixed) - seeds 30 down to 15 (16 runs)
::  Output: output\dtgpn\{log,pkl,csv}\MNIST-LT-200\
::  Place this file in the project root (same dir as dtgpn).
::  EDIT PYTHON_EXE below if your env path differs.
:: ======================================================================
set "PYTHON_EXE=D:\softwaredownload\Anaconda\envs\GP\python.exe"

set "METHOD=dtgpn"
set "DATASET=mnist"
set "LABEL=MNIST-LT-200"
set "LTMODE=fixed"
set "SEED_START=30"
set "SEED_END=15"

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
::  Runs seeds SEED_START .. SEED_END in descending order with .done resume.
:: ======================================================================
:run_task
set "OUTBASE=%~dp0output\%METHOD%\"

mkdir "%OUTBASE%log\%LABEL%" 2>nul
mkdir "%OUTBASE%pkl\%LABEL%" 2>nul
mkdir "%OUTBASE%csv\%LABEL%" 2>nul

for /l %%i in (%SEED_START%,-1,%SEED_END%) do (
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
