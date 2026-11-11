@echo off
setlocal enabledelayedexpansion

:: ======================================================================
::  CarDD (6 classes, original class distribution, no long-tail sampling)
::  Runs seeds 1..5 for each method, with .done resume.
::  Place this file in the project root (same dir as the six method folders).
::  EDIT PYTHON_EXE below if your env path differs.
:: ======================================================================
set "PYTHON_EXE=E:\Code\py311\python.exe"

echo [Info] PYTHON_EXE=%PYTHON_EXE%
if not exist "%PYTHON_EXE%" (
    echo [Fatal] PYTHON_EXE not found: "%PYTHON_EXE%"
    echo         Please edit the PYTHON_EXE line at the top of this script.
    exit /b 2
)

for %%M in (FGP GP-FR GP-FR-recall MEGPE_contrast SLGP-LT-new dtgpn) do (
    if not exist "%~dp0%%M\" (
        echo [Fatal] method folder not found: "%~dp0%%M"
        exit /b 2
    )
    echo.
    echo ============================================================
    echo  [CarDD] method: %%M
    echo ============================================================
    pushd "%~dp0%%M"
    call :run_task
    set "RC=!errorlevel!"
    popd
    if "!RC!"=="130" goto :abort
)

echo.
echo [ALL DONE - CarDD]
exit /b 0

:abort
echo.
echo [ABORT] Ctrl+C detected, all runs stopped.
exit /b 130

:: ======================================================================
::  run_task
::  Runs seeds 1..5 of CarDD inside the current method folder.
::  CarDD: 6 classes, raw class counts (no imbalance_MRBand), --disable_imb_test
:: ======================================================================
:run_task
set "DATASET=CarDD"
set "LABEL=CarDD"

:: 输出统一放到 根目录\output\<方法名>\log|pkl|csv
for %%A in ("%CD%") do set "METHOD=%%~nxA"
set "OUTBASE=%~dp0output\!METHOD!\"

mkdir "!OUTBASE!log\!LABEL!" 2>nul
mkdir "!OUTBASE!pkl\!LABEL!" 2>nul
mkdir "!OUTBASE!csv\!LABEL!" 2>nul

for /l %%i in (1,1,5) do (
    set "LOGFILE=!OUTBASE!log\!LABEL!\!DATASET!_%%i.log"
    set "PKLFILE=!OUTBASE!pkl\!LABEL!\!DATASET!_%%i.pkl"
    set "CSVFILE=!OUTBASE!csv\!LABEL!\!DATASET!_%%i.csv"
    set "DONEFILE=!OUTBASE!pkl\!LABEL!\!DATASET!_%%i.done"

    if exist "!DONEFILE!" (
        echo [Skip] !DATASET! seed %%i already completed.
    ) else (
        echo [Start/Resume] !DATASET! seed %%i ...
        "%PYTHON_EXE%" monitor_subprocess.py --save "!CSVFILE!" --include-children "%PYTHON_EXE%" main.py --log_file "!LOGFILE!" --pkl_file "!PKLFILE!" --dataset !DATASET! --randomSeeds %%i --disable_imb_test --population 100 --generation 50 --num_processes 16

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
