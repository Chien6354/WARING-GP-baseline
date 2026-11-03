@echo off
setlocal enabledelayedexpansion

:: ======================================================================
::  Machine C - EuroSAT clean + rerun (standalone script)
::  Place this file in the project root (same dir as GP-FR-recall, GP-FR).
::  EDIT PYTHON_EXE below if your env path differs.
:: ======================================================================
set "PYTHON_EXE=D:\software\miniconda\envs\FlexGP\python.exe"

:: ---------------- GP-FR-recall (EuroSAT) ----------------
pushd "%~dp0GP-FR-recall"
call :clean_task "EuroSAT-LT-400"
call :run_task "EuroSAT" "EuroSAT-LT-400" "exp"
popd

:: ---------------- GP-FR (EuroSAT) ----------------
pushd "%~dp0GP-FR"
call :clean_task "EuroSAT-LT-400"
call :run_task "EuroSAT" "EuroSAT-LT-400" "exp"
popd

echo.
echo [ALL DONE - Machine C EuroSAT]
exit /b 0

:: ======================================================================
::  clean_task  <output-dir-label>
::  Deletes log/pkl/csv of the given label inside the current folder.
:: ======================================================================
:clean_task
set "LABEL=%~1"
if exist "log\!LABEL!" rd /s /q "log\!LABEL!"
if exist "pkl\!LABEL!" rd /s /q "pkl\!LABEL!"
if exist "csv\!LABEL!" rd /s /q "csv\!LABEL!"
echo [Clean] removed output for !LABEL!
exit /b 0

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

        if !errorlevel! equ 0 (
            echo [Success] !DATASET! seed %%i finished.
            echo. > "!DONEFILE!"
        ) else (
            echo [Error/Interrupted] !DATASET! seed %%i not finished. Will resume next time.
        )
    )
)
exit /b 0
