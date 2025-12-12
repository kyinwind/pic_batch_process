@echo off
chcp 65001 >nul 2>&1
cls

:: ==============================================
::          激活 Python base 环境并运行 go.py
:: ==============================================
echo 执行目录：%cd%
echo 执行时间：%date% %time%
echo ==============================================
echo.

:: 设置 conda 安装路径（根据你实际情况修改）
set "CONDA_PATH=C:\ProgramData\miniconda3"

:: 检查 conda 是否存在
if not exist "%CONDA_PATH%\Scripts\activate.bat" (
    echo 错误：未找到 Anaconda/Miniconda！
    echo 请检查 CONDA_PATH 是否正确
    echo.
    pause
    exit /b 1
)

:: 激活 base
echo 正在激活 base 环境...
call "%CONDA_PATH%\Scripts\activate.bat" base
echo base 环境激活成功！
echo.

:: 执行 go.py
echo 正在执行 go.py...
echo ----------------------------------------------
python go.py

if %errorlevel% equ 0 (
    echo.
    echo 执行成功！
) else (
    echo.
    echo 执行失败，请检查错误信息
)
echo ----------------------------------------------
echo.
pause
