@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: 分布式实验脚本 [电脑 3/3]
::   任务2: SLGP-LT-new switch_ratio 敏感性 (0.25 / 0.75, cifar10 fixed)
::   任务3(部分): SLGP-LT-new / dtgpn / GP-FR-recall x MRD
::   每组 10 个 seed: 1-10; 断点续跑 (.done)
::   注意: 整个 WARING-GP-baseline 目录(含 datasets)可放本机任意路径, 脚本自动定位
::   提示: 任务2 fixed 模式与其他实验一样直接从 datasets\CIFAR-10 加载, 无需项目内 data
:: ============================================================

set "PYTHON_EXE=D:\software\Anaconda\envs\GP\python.exe"
set "BASE=%~dp0"

set "map_0.25=025"
set "map_0.75=075"

echo ============================================================
echo  [PC3] Task 2: SLGP-LT-new switch_ratio sensitivity on cifar10
echo ============================================================
for %%R in (0.25 0.75) do (
    set "SUF=!map_%%R!"
    call :run_config "SLGP-LT-new" "cifar10" "CIFAR10-LT-200-switch!SUF!" "--switch_ratio %%R --disable_imb_test --population 100 --generation 50 --num_processes 10"
)

echo ============================================================
echo  [PC3] Task 3 (part): 3 methods x MRD
echo ============================================================
for %%P in (SLGP-LT-new dtgpn GP-FR-recall) do (
    call :run_config "%%P" "MRD" "MRD-LT-200" "--disable_imb_test --population 100 --generation 50 --num_processes 10"
)

echo ============================================================
echo  [PC3] All assigned experiments finished!
echo ============================================================
endlocal
pause
goto :eof

:: ------------------------------------------------------------
:: 子过程: 在指定项目目录下跑一个 (数据集 x 配置) 的 10 个 seed
::   %1 = 项目目录名   %2 = 数据集名   %3 = 输出子目录名   %4 = 额外参数(可为空)
:: ------------------------------------------------------------
:run_config
set "PROJ=%~1"
set "DS=%~2"
set "DIRNAME=%~3"
set "EXTRA=%~4"

echo.
echo ########## %PROJ% ^| dataset=%DS% ^| out=%DIRNAME% ^| extra=%EXTRA% ##########

cd /d "%BASE%%PROJ%"
if errorlevel 1 (
    echo [Error] Project folder not found: %BASE%%PROJ%
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
