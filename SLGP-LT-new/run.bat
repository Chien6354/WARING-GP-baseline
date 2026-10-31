@echo off
setlocal enabledelayedexpansion

:: 定义数据集到目录名称的映射 (模拟字典)
set "map_mnist=MB-LT-200"
set "map_cifar10=CIFAR10-LT-200"
set "map_fashion_mnist=Fashion-MNIST-LT-200"
set "map_STL=STL-LT-200"
set "map_SVHN=SVHN-LT-200"
set "map_EuroSAT=EuroSAT-LT-200"

:: 新的数据集列表
set "datasets=SVHN EuroSAT STL "

for %%d in (%datasets%) do (
    :: 获取当前数据集对应的目录名称
    set "DIR_NAME=!map_%%d!"

    echo ====================================================
    echo Checking dataset: %%d ^(Folder: !DIR_NAME!^)
    echo ====================================================

    :: 为当前数据集创建对应的子目录 (2>nul 用于屏蔽目录已存在时的提示)
    mkdir "log\!DIR_NAME!" 2>nul
    mkdir "pkl\!DIR_NAME!" 2>nul
    mkdir "csv\!DIR_NAME!" 2>nul

    :: 统一从 1 循环到 30
    for /l %%i in (1,1,30) do (
        set "LOGFILE=log\!DIR_NAME!\%%d_%%i.log"
        set "PKLFILE=pkl\!DIR_NAME!\%%d_%%i.pkl"
        set "CSVFILE=csv\!DIR_NAME!\%%d_%%i.csv"
        set "DONEFILE=pkl\!DIR_NAME!\%%d_%%i.done"

        :: 检查 DONE 文件是否存在，如果存在说明该 run 已经完整跑完，跳过
        if exist "!DONEFILE!" (
            echo [Skip] %%d run %%i already completed ^(Found !DONEFILE!^).
        ) else (
            :: 如果没有 done 文件，但存在 pkl 文件，说明是中断的任务，准备续训
            if exist "!PKLFILE!" (
                echo [Resume] Found incomplete checkpoint, resuming %%d run %%i...
            ) else (
                echo [Start] Starting fresh run for %%d run %%i...
            )

            :: 执行 Python 任务
            python monitor_subprocess.py --save "!CSVFILE!" --include-children python main.py --log_file "!LOGFILE!" --pkl_file "!PKLFILE!" --dataset %%d --randomSeeds %%i

            :: 检查上一条 Python 命令是否成功结束（没有报错或被手动中断）
            if !errorlevel! equ 0 (
                echo [Success] Run %%i finished successfully. Marking as done.
                :: 创建一个空的 .done 文件作为完成标记
                echo. > "!DONEFILE!"
            ) else (
                echo [Interrupted/Error] Run %%i did not finish properly. Will resume next time.
                :: 如果你希望遇到错误时直接停止整个 bat 脚本，可以取消注释下面这行：
                :: exit /b !errorlevel!
            )
        )
    )
)

echo All tasks finished!
endlocal
pause
