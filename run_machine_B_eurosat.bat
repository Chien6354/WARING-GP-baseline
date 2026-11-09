@echo off
setlocal enabledelayedexpansion

:: ======================================================================
::  Machine B - EuroSAT resume only (standalone script)
::  Place this file in the project root (same dir as SLGP-LT-new, FGP).
::  EDIT PYTHON_EXE below if your env path differs.
:: ======================================================================
set "PYTHON_EXE=D:\software\Anaconda\envs\GP\python.exe"

:: ---------------- SLGP-LT-new (EuroSAT) ----------------
pushd "%~dp0SLGP-LT-new"
call :run_task "EuroSAT" "EuroSAT-LT-400" "exp"
if "!errorlevel!"=="130" goto :abort
popd

:: ---------------- FGP (EuroSAT) ----------------
pushd "%~dp0FGP"
call :run_task "EuroSAT" "EuroSAT-LT-400" "exp"
if "!errorlevel!"=="130" goto :abort
popd

echo.
echo [ALL DONE - Machine B EuroSAT]
exit /b 0

:abort
echo.
echo [ABORT] Ctrl+C detected, all runs stopped.
exit /b 130

:: ======================================================================
::  run_task  <dataset-token>  <output-dir-label>  <lt_mode>
::  Runs seeds 1..30 with .done resume inside the current method folder.
:: ======================================================================
:run_task
set "DATASET=%~1"
set "LABEL=%~2"
set "LTMODE=%~3"

mkdir "log\!LABEL!" 2>nul
mkdir "pkl\!LABEL!" 2>nul
mkdir "csv\!LABEL!" 2>nul

for /l %%i in (1,1,30) do (
    set "LOGFILE=log\!LABEL!\!DATASET!_%%i.log"
    set "PKLFILE=pkl\!LABEL!\!DATASET!_%%i.pkl"
    set "CSVFILE=csv\!LABEL!\!DATASET!_%%i.csv"
    set "DONEFILE=pkl\!LABEL!\!DATASET!_%%i.done"

    if exist "!DONEFILE!" (
        echo [Skip] !DATASET! seed %%i already completed.
    ) else (
        echo [Start/Resume] !DATASET! seed %%i ...
        "%PYTHON_EXE%" monitor_subprocess.py --save "!CSVFILE!" --include-children "%PYTHON_EXE%" main.py --log_file "!LOGFILE!" --pkl_file "!PKLFILE!" --dataset !DATASET! --randomSeeds %%i --lt_mode !LTMODE! --disable_imb_test --population 100 --generation 50 --num_processes 10

        if !errorlevel! equ 130 (
            echo [ABORT] Ctrl+C detected, stopping remaining runs.
            exit /b 130
        )
        if !errorlevel! equ 0 (
            echo [Success] !DATASET! seed %%i finished.
            echo. > "!DONEFILE!"
        ) else (
            echo [Error/Interrupted] !DATASET! seed %%i not finished. Will resume next time.
        )
    )
)
exit /b 0
