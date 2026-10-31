@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: ?????????????????????????��?????? 30 ?? seed: 1-30??
::   ????1: 6 ?????? x cifar10 (exp ??��, ?400/��20/????20)
::   ????2: SLGP-LT-new ?��??????????? (switch_ratio = 0.25/0.5/0.75, ???????? cifar10)
::   ????3: SLGP-LT-new / dtgpn / GP-FR-recall x fashion_mnist + MRD (????????)
:: ???: ?????????????? --disable_imb_test ?????????????????? test ????
:: ???????: ??? run ????????? .done ??????????? run ?????????
:: ============================================================

set "PYTHON_EXE=D:\software\Anaconda\envs\GP\python.exe"
set "BASE=F:\WARING-GP-baseline"

:: switch_ratio ????????????
set "map_0.25=025"
@REM set "map_0.5=050"
set "map_0.75=075"

echo ============================================================
echo  Task 1: 6 methods x cifar10 (exp LT: head=400, tail=20)
echo ============================================================
for %%P in (FGP GP-FR GP-FR-recall MEGPE_contrast SLGP-LT-new dtgpn) do (
    call :run_config "%%P" "cifar10" "CIFAR10-LT-400" "--lt_mode exp --disable_imb_test --population 100 --generation 50 --num_processes 10"
)

echo ============================================================
echo  Task 2: SLGP-LT-new switch_ratio sensitivity on cifar10
echo ============================================================
for %%R in (0.25 0.75) do (
    set "SUF=!map_%%R!"
    call :run_config "SLGP-LT-new" "cifar10" "CIFAR10-LT-200-switch!SUF!" "--switch_ratio %%R --disable_imb_test --population 100 --generation 50 --num_processes 10"
)

echo ============================================================
echo  Task 3: 3 methods x fashion_mnist + MRD (original LT)
echo ============================================================
for %%P in (SLGP-LT-new dtgpn GP-FR-recall) do (
    call :run_config "%%P" "fashion_mnist" "Fashion-MNIST-LT-200" "--disable_imb_test --population 100 --generation 50 --num_processes 10"
    call :run_config "%%P" "MRD" "MRD-LT-200" "--disable_imb_test --population 100 --generation 50 --num_processes 10"
)

echo ============================================================
echo  All experiments finished!
echo ============================================================
endlocal
pause
goto :eof

:: ------------------------------------------------------------
:: ?????: ????????????????? (????? x ????) ?? 30 ?? seed
::   %1 = ???????   %2 = ???????   %3 = ?????????   %4 = ???????(?????)
:: ------------------------------------------------------------
:run_config
set "PROJ=%~1"
set "DS=%~2"
set "DIRNAME=%~3"
set "EXTRA=%~4"

echo.
echo ########## %PROJ% ^| dataset=%DS% ^| out=%DIRNAME% ^| extra=%EXTRA% ##########

cd /d "%BASE%\%PROJ%"
if errorlevel 1 (
    echo [Error] Project folder not found: %BASE%\%PROJ%
    cd /d "%BASE%"
    goto :eof
)

mkdir "log\%DIRNAME%" 2>nul
mkdir "pkl\%DIRNAME%" 2>nul
mkdir "csv\%DIRNAME%" 2>nul

for /l %%i in (1,1,10) do (
    set "LOGFILE=log\%DIRNAME%\%DS%_%%i.log"
    set "PKLFILE=pkl\%DIRNAME%\%DS%_%%i.pkl"
    set "CSVFILE=csv\%DIRNAME%\%DS%_%%i.csv"
    set "DONEFILE=pkl\%DIRNAME%\%DS%_%%i.done"

    if exist "!DONEFILE!" (
        echo [Skip] %PROJ% %DS% run %%i already completed.
    ) else (
        if exist "!PKLFILE!" (
            echo [Resume] %PROJ% %DS% run %%i from checkpoint...
        ) else (
            echo [Start] %PROJ% %DS% run %%i...
        )

        "%PYTHON_EXE%" monitor_subprocess.py --save "!CSVFILE!" --include-children "%PYTHON_EXE%" main.py --log_file "!LOGFILE!" --pkl_file "!PKLFILE!" --dataset %DS% --randomSeeds %%i %EXTRA%

        if !errorlevel! equ 0 (
            echo [Success] %PROJ% %DS% run %%i finished.
            echo. > "!DONEFILE!"
        ) else (
            echo [Interrupted/Error] %PROJ% %DS% run %%i did not finish. Will resume next time.
        )
    )
)

cd /d "%BASE%"
goto :eof
