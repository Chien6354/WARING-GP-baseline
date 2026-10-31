@echo off
:: 设置 Python 解释器路径
set PYTHON_EXE="D:\Code\GP_code\mulGP_imbalance\venv\Scripts\python.exe"

:: 创建日志目录（如果不存在）
mkdir log 2>nul

:: 定义 dataset 的取值范围
set "datasets=STL"

:: 循环 dataset 和 augment 参数
for %%d in (%datasets%) do (
    for %%a in (False) do (
        echo Running with dataset=%%d, augment=%%a
        :: 内部循环：运行 30 次
        for /l %%i in (26,1,30) do (
            echo Running: %PYTHON_EXE% COGP_main.py --log_file "log/COGP_%%d_%%a_%%i.log" --pkl_file "log/COGP_%%d_%%a_%%i.pkl" --dataset %%d --augment %%a  --randomSeeds %%i
            %PYTHON_EXE% COGP_main.py --log_file "log/COGP_%%d_%%a_%%i.log" --pkl_file "log/COGP_%%d_%%a_%%i.pkl" --dataset %%d --augment %%a --randomSeeds %%i
        )
    )
)

pause