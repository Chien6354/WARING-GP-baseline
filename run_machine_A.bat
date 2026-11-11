@echo off
setlocal enabledelayedexpansion

:: ======================================================================
::  Machine A (FAST machine)
::  Place this file in the project root (same dir as SLGP-LT-new, dtgpn,
::  GP-FR-recall, FGP, GP-FR, MEGPE_contrast).
::  EDIT PYTHON_EXE below if your env path differs.
:: ======================================================================
set "PYTHON_EXE=D:\softwaredownload\Anaconda\envs\GP\python.exe"

:: ---------------- dtgpn (slowest -> run on fast machine) ----------------
pushd "%~dp0dtgpn"
call :run_task "mnist" "MNIST-LT-200" "fixed"
if "!errorlevel!"=="130" goto :abort
call :run_task "EuroSAT" "EuroSAT-LT-400" "exp"
if "!errorlevel!"=="130" goto :abort
popd

:: ---------------- MEGPE_contrast ----------------
pushd "%~dp0MEGPE_contrast"
call :run_task "EuroSAT" "EuroSAT-LT-400" "exp"
if "!errorlevel!"=="130" goto :abort
popd

echo.
echo [ALL DONE - Machine A]
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

:: 输出统一放到 根目录\output\<方法名>\log|pkl|csv
for %%A in ("%CD%") do set "METHOD=%%~nxA"
set "OUTBASE=%~dp0output\!METHOD!\"

mkdir "!OUTBASE!log\!LABEL!" 2>nul
mkdir "!OUTBASE!pkl\!LABEL!" 2>nul
mkdir "!OUTBASE!csv\!LABEL!" 2>nul

for /l %%i in (1,1,30) do (
    set "LOGFILE=!OUTBASE!log\!LABEL!\!DATASET!_%%i.log"
    set "PKLFILE=!OUTBASE!pkl\!LABEL!\!DATASET!_%%i.pkl"
    set "CSVFILE=!OUTBASE!csv\!LABEL!\!DATASET!_%%i.csv"
    set "DONEFILE=!OUTBASE!pkl\!LABEL!\!DATASET!_%%i.done"

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