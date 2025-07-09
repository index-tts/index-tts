@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: 环境检查 - Conda
echo 正在检查Conda安装...
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo [conda已安装]
) else (
    echo 错误：未检测到conda安装
    echo 请先安装Anaconda或Miniconda并添加至系统路径
    pause
    exit /b 1
)

:: 检查CUDA版本并设置PyTorch版本
echo 正在检测CUDA版本...
set "pytorchversion=128"  :: 默认值
for /f "tokens=2 delims=:" %%a in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do (
    for /f "tokens=1-2 delims=." %%b in ("%%a") do (
        set /a "cuda_major=%%b, cuda_minor=%%c"
        set "cuda_version=%%b.%%c"
    )
)

if defined cuda_version (
    echo 检测到CUDA版本: !cuda_version!
    :: 版本比较逻辑
    if !cuda_major! equ 12 (
        if !cuda_minor! lss 6 (
            set "pytorchversion=118"
        ) else if !cuda_minor! lss 8 (
            set "pytorchversion=126"
        ) else (
            set "pytorchversion=128"
        )
    ) else if !cuda_major! lss 12 (
        set "pytorchversion=118"
    )
) else (
    echo 警告：未检测到NVIDIA驱动，将使用CPU模式安装
    set "pytorchversion=cpu"
)

echo 将使用PyTorch版本: !pytorchversion!

:: 用户确认
echo.
echo 注意：安装需要约11GB磁盘空间
set /p confirm="是否继续安装? (y/n): "
if /i "!confirm!" neq "y" (
    echo 安装已取消
    exit /b 0
)

:: 创建conda环境（带重试机制）
set "env_created=0"
set "max_retries=3"
set "retry_count=0"

:create_env
echo 正在创建conda环境(index-tts)...
conda create -n index-tts python=3.10 -y
if %errorlevel% equ 0 (
    set "env_created=1"
    echo 环境创建成功
) else (
    set /a "retry_count+=1"
    if !retry_count! geq !max_retries! (
        echo 错误：环境创建失败 - 网络不可达或conda错误
        pause
        exit /b 1
    )
    echo 环境创建失败，10秒后重试(!retry_count!/!max_retries!)...
    timeout /t 10 /nobreak >nul
    goto create_env
)

:: 激活环境
call conda.bat activate index-tts
if %errorlevel% neq 0 (
    echo 错误：无法激活index-tts环境
    pause
    exit /b 1
)

:: 安装依赖函数（带重试机制）
:install_with_retry
set "cmd=%~1"
set "desc=%~2"
set "retry_count=0"

:install_retry
echo 正在安装: !desc!
%cmd%
if %errorlevel% equ 0 (
    echo !desc! 安装成功
    goto :eof
) else (
    set /a "retry_count+=1"
    if !retry_count! geq %max_retries% (
        echo 错误：!desc! 安装失败 - 网络不可达
        pause
        exit /b 1
    )
    echo !desc! 安装失败，10秒后重试(!retry_count!/%max_retries%)...
    timeout /t 10 /nobreak >nul
    goto install_retry
)

:: 安装依赖项
call :install_with_retry "conda install -c conda-forge ffmpeg -y" "FFmpeg"
call :install_with_retry "conda install -c conda-forge pynini==2.1.6 -y" "Pynini"
call :install_with_retry "pip install -r requirements.txt" "基础依赖"
call :install_with_retry "pip install WeTextProcessing==1.0.3" "WeTextProcessing"
call :install_with_retry "pip install -e .[webui]" "WebUI扩展"
call :install_with_retry "pip uninstall torch torchaudio -y" "卸载旧版PyTorch"

:: 安装PyTorch
if "!pytorchversion!"=="cpu" (
    call :install_with_retry "pip install torch torchvision torchaudio" "CPU版PyTorch"
) else (
    call :install_with_retry "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu!pytorchversion!" "CUDA版PyTorch"
)

call :install_with_retry "pip install modelscope" "ModelScope"

:: 下载模型
echo 正在下载IndexTTS模型...
modelscope download --model IndexTeam/IndexTTS-1.5 --local_dir ./checkpoints/
if %errorlevel% neq 0 (
    echo 警告：模型下载失败，请手动下载
)

:: 完成提示
echo.
echo =========================================
echo  安装成功！请使用以下命令启动:
echo     conda activate index-tts
echo     python webui.py
echo =========================================
pause
exit /b 0