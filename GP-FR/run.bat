@echo off
setlocal enabledelayedexpansion

:: 请根据实际路径设置 Python 可执行
set PYTHON_EXE=D:\software\Anaconda\envs\GP\python.exe

:: 创建目录
mkdir log 2>nul
mkdir pkl 2>nul
mkdir csv 2>nul

:: 数据集列表
set "datasets=SVHN EuroSAT "

for %%d in (%datasets%) do (
    echo Running with dataset=%%d
    for /l %%i in (1,1,30) do (
        set "LOGFILE=log\%%d_%%i.log"
        set "PKLFILE=pkl\%%d_%%i.pkl"
        set "CSVFILE=csv\%%d_%%i.csv"

        echo Running iteration %%i: logging to !LOGFILE! and CSV to !CSVFILE!

        :: 调用 monitor 脚本（注意替换 monitor_subprocess.py 为你的文件名）
        "%PYTHON_EXE%" monitor_subprocess.py --save "!CSVFILE!" --include-children "%PYTHON_EXE%" main.py --log_file "!LOGFILE!" --pkl_file "!PKLFILE!" --dataset %%d --randomSeeds %%i

        :: 如果 monitor 已实现自动去掉 leading '--'，也可以不传 '--'，例如：
        :: "%PYTHON_EXE%" monitor_subprocess.py --save "!CSVFILE!" --interval 1 COGP_main.py --log_file "!LOGFILE!" ...
    )
)

endlocal
pause
