@echo off
setlocal enabledelayedexpansion

:: ============================================================================
::  collect_results.bat  <PC编号>
::  在每台实验机的"项目根目录"运行一次, 把本机 6 个方法的实验结果收集到
::  collected_results\PC<编号> 目录, 之后把这个目录拷回主分析机即可.
::
::  用法: collect_results.bat 1     (产生 collected_results\PC1)
::         collect_results.bat 2     (产生 collected_results\PC2)
::         collect_results.bat 3     (产生 collected_results\PC3)
:: ============================================================================

if "%~1"=="" (
    echo [ERROR] 缺少本机编号参数.
    echo 用法: collect_results.bat ^<PC编号^>   例如: collect_results.bat 1
    exit /b 1
)

set "PC=%~1"
set "DEST=%~dp0collected_results\PC%PC%"

:: 6 个 GP 方法目录
set "METHODS=SLGP-LT-new dtgpn FGP GP-FR GP-FR-recall MEGPE_contrast"

mkdir "%DEST%" 2>nul

for %%m in (%METHODS%) do (
    if exist "%~dp0%%m\log" (
        echo [collecting] %%m
        xcopy "%~dp0%%m\log" "!DEST!\%%m\log\" /E /I /Y >nul
        xcopy "%~dp0%%m\csv" "!DEST!\%%m\csv\" /E /I /Y >nul
        :: pkl 是断点续跑用的临时文件, 体积较大; 若不需要可删掉下面这行
        xcopy "%~dp0%%m\pkl" "!DEST!\%%m\pkl\" /E /I /Y >nul
    ) else (
        echo [skip] %%m ^(本机无该方法结果^)
    )
)

echo.
echo [DONE] 结果已收集到: !DEST!
echo 请把整个 collected_results\PC%PC% 目录拷回主分析机.
endlocal
pause