@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: e???????: §³???????????????????? (?????????????????)
::   ???: ?? seed=1, population=10, num_processes=10
::   ????: SLGP-LT-new=6 (??????????), ??????=3
::   ????????? TEST- ??, ???????????, ?????????????
:: ============================================================

set "PYTHON_EXE=D:\software\Anaconda\envs\GP\python.exe"
set "BASE=F:\WARING-GP-baseline"

set "map_0.25=025"
set "map_0.5=050"
set "map_0.75=075"

echo ============================================================
echo  [TEST] Task 1: 6 methods x cifar10 (exp LT)
echo ============================================================
for %%P in (SLGP-LT-new dtgpn FGP GP-FR GP-FR-recall MEGPE_contrast) do (
    if "%%P"=="SLGP-LT-new" (
        call :run_config "%%P" "cifar10" "TEST-CIFAR10-LT-400" "--lt_mode exp --disable_imb_test --population 10 --generation 6 --num_processes 10"
    ) else (
        call :run_config "%%P" "cifar10" "TEST-CIFAR10-LT-400" "--lt_mode exp --disable_imb_test --population 10 --generation 3 --num_processes 10"
    )
)

echo ============================================================
echo  [TEST] Task 2: SLGP-LT-new switch_ratio sensitivity on cifar10
echo ============================================================
for %%R in (0.25 0.5 0.75) do (
    set "SUF=!map_%%R!"
    call :run_config "SLGP-LT-new" "cifar10" "TEST-CIFAR10-LT-200-switch!SUF!" "--switch_ratio %%R --disable_imb_test --population 10 --generation 6 --num_processes 10"
)

echo ============================================================
echo  [TEST] Task 3: 3 methods x fashion_mnist + MRD
echo ============================================================
for %%P in (SLGP-LT-new dtgpn GP-FR-recall) do (
    if "%%P"=="SLGP-LT-new" (
        call :run_config "%%P" "fashion_mnist" "TEST-Fashion-MNIST-LT-200" "--disable_imb_test --population 10 --generation 6 --num_processes 10"
        call :run_config "%%P" "MRD" "TEST-MRD-LT-200" "--disable_imb_test --population 10 --generation 6 --num_processes 10"
    ) else (
        call :run_config "%%P" "fashion_mnist" "TEST-Fashion-MNIST-LT-200" "--disable_imb_test --population 10 --generation 3 --num_processes 10"
        call :run_config "%%P" "MRD" "TEST-MRD-LT-200" "--disable_imb_test --population 10 --generation 3 --num_processes 10"
    )
)

echo ============================================================
echo  [TEST] All smoke tests finished!
echo ============================================================
endlocal
pause
goto :eof

:: ------------------------------------------------------------
:: ?????: ???????????, ????? seed=1
:: ------------------------------------------------------------
:run_config
set "PROJ=%~1"
set "DS=%~2"
set "DIRNAME=%~3"
set "EXTRA=%~4"

echo.
echo ########## [TEST] %PROJ% ^| dataset=%DS% ^| out=%DIRNAME% ##########

cd /d "%BASE%\%PROJ%"
if errorlevel 1 (
    echo [Error] Project folder not found: %BASE%\%PROJ%
    cd /d "%BASE%"
    goto :eof
)

mkdir "log\%DIRNAME%" 2>nul
mkdir "pkl\%DIRNAME%" 2>nul
mkdir "csv\%DIRNAME%" 2>nul

set "LOGFILE=log\%DIRNAME%\%DS%_1.log"
set "PKLFILE=pkl\%DIRNAME%\%DS%_1.pkl"
set "CSVFILE=csv\%DIRNAME%\%DS%_1.csv"
set "DONEFILE=pkl\%DIRNAME%\%DS%_1.done"

if exist "%DONEFILE%" (
    echo [Skip] %PROJ% %DS% test run already completed.
) else (
    echo [Start] %PROJ% %DS% test run, seed=1 ...
    "%PYTHON_EXE%" monitor_subprocess.py --save "%CSVFILE%" --include-children "%PYTHON_EXE%" main.py --log_file "%LOGFILE%" --pkl_file "%PKLFILE%" --dataset %DS% --randomSeeds 1 %EXTRA%
    if !errorlevel! equ 0 (
        echo [Success] %PROJ% %DS% test run finished.
        echo. > "%DONEFILE%"
    ) else (
        echo [FAILED] %PROJ% %DS% test run did not finish. Check %LOGFILE%
    )
)

cd /d "%BASE%"
goto :eof
