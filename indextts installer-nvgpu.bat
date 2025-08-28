@echo off & title IndexTTS 安装脚本
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ==================== 配置参数 ====================
set "CONDA_ENV_NAME=index-tts"
set "PYTHON_VERSION=3.10"
set "MAX_RETRIES=3"
set "DEFAULT_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

:: ==================== 环境检查 ====================
echo index-tts运行需要conda和CUDA，请确保运行环境已正确安装
echo .
echo .
echo 正在检查系统环境...

:: 检查Conda安装
echo 正在检查Conda安装...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ================================
    echo [×] 未找到conda
    echo 请先安装Anaconda或Miniconda并添加至系统路径
    echo .
    echo 请前往官网进行安装：
    echo https://www.anaconda.com/download/success
    echo ================================
    pause
    exit /b 1
)
echo [√] conda已安装

:: 检查CUDA Toolkit安装
echo 正在检查CUDA Toolkit安装情况...
set "cuda_detected=0"
set "cuda_path="
set "latest_cuda_version="

:: 方法1：检查环境变量
if defined CUDA_PATH (
    set "cuda_path=!CUDA_PATH!"
    echo [√] 找到环境变量 CUDA_PATH: !cuda_path!
    set cuda_detected=1
) else (
    echo [×] 未找到环境变量 CUDA_PATH
)

:: 方法2：扫描默认安装目录
if exist "!DEFAULT_CUDA_PATH!" (
    echo 正在扫描默认安装目录: !DEFAULT_CUDA_PATH!
    
    set "max_ver=0"
    set "version_count=0"
    
    for /f "tokens=*" %%i in ('dir /b /ad "!DEFAULT_CUDA_PATH!\v*" 2^>nul') do (
        set /a version_count+=1
        set "ver_dir=%%i"
        
        :: 提取版本号并转换为可比较的数字
        set "ver_str=!ver_dir:~1!"
        set "ver_num=!ver_str:.=!"
        set /a ver_num=!ver_num! 2>nul
        
        if !ver_num! gtr !max_ver! (
            set max_ver=!ver_num!
            set "latest_cuda_version=!ver_dir!"
        )
        echo  发现版本: %%i
    )
    
    if !version_count! gtr 0 (
        echo.
        echo 共发现 !version_count! 个CUDA版本，选择最高版本 !latest_cuda_version! 进行验证
        
        if !cuda_detected! equ 0 (
            set "cuda_path=!DEFAULT_CUDA_PATH!\!latest_cuda_version!"
            set cuda_detected=1
        )
    ) else (
        echo [×] 默认路径下未找到任何CUDA版本
    )
)

:: 验证CUDA安装
if !cuda_detected! equ 1 (
    echo.
    echo 正在验证CUDA安装完整性...
    echo 检查路径: !cuda_path!
    
    set "file_valid=1"
    
    :: 检查关键文件
    if not exist "!cuda_path!\bin\nvcc.exe" (
        echo [×] 缺少关键文件: nvcc.exe
        set file_valid=0
    ) else (
        echo [√] 找到编译器: nvcc.exe
    )
    
    dir /b "!cuda_path!\bin\cudart_*.dll" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [×] 缺少关键文件: cudart_*.dll
        set file_valid=0
    ) else (
        for %%f in ("!cuda_path!\bin\cudart_*.dll") do echo [√] 找到运行时库: %%~nxf
    )
    
    dir /b "!cuda_path!\bin\cublas64_*.dll" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [×] 缺少关键文件: cublas64_*.dll
        set file_valid=0
    ) else (
        for %%f in ("!cuda_path!\bin\cublas64_*.dll") do echo [√] 找到cuBLAS库: %%~nxf
    )
    
    if !file_valid! equ 0 set cuda_detected=0
)

:: 显示CUDA检测结果
echo.
if !cuda_detected! equ 1 (
    echo ================================
    echo CUDA Toolkit 已正确安装
    echo 路径: !cuda_path!
    echo ================================
) else (
    echo ================================
    echo 警告: 未检测到有效的CUDA安装
    echo ================================
    echo 可能原因:
    echo 1. 未安装CUDA Toolkit
    echo 2. 安装后未重启系统
    echo 3. 自定义安装路径未设置环境变量
    echo 4. 关键文件损坏或缺失
    echo.
    echo 请从NVIDIA官网下载安装:
    echo https://developer.nvidia.com/cuda-toolkit
    echo.
)

:: ==================== 确定PyTorch版本 ====================
set "pytorch_version=cu128"  :: 默认值

if defined latest_cuda_version (
    echo 检测到CUDA版本: !latest_cuda_version!
    
    :: 提取版本号
    set "version_part=!latest_cuda_version:v=!"
    for /f "tokens=1,2 delims=." %%a in ("!version_part!") do (
        set "cuda_major=%%a"
        set "cuda_minor=%%b"
    )
    
    :: 版本匹配逻辑
    if !cuda_major! equ 12 (
        if !cuda_minor! lss 6 (
            set "pytorch_version=cu118"
        ) else if !cuda_minor! lss 8 (
            set "pytorch_version=cu126"
        ) else (
            set "pytorch_version=cu128"
        )
    ) else if !cuda_major! lss 12 (
        set "pytorch_version=cu118"
    )
) else (
    echo 警告：未检测到NVIDIA驱动，将使用CPU模式安装
    set "pytorch_version=cpu"
)

echo 将使用PyTorch版本: !pytorch_version!

:: ==================== 用户确认 ====================
echo.
echo 注意：安装需要约11GB磁盘空间
set /p confirm="是否继续安装? (y/n): "
if /i "!confirm!" neq "y" (
    echo 安装已取消
    exit /b 0
)

:: ==================== 安装函数 ====================
:install_with_retry
set "install_cmd=%~1"
set "install_desc=%~2"
set "retry_count=0"

:retry_loop
echo 正在安装: !install_desc!
%install_cmd%
if %errorlevel% equ 0 (
    echo !install_desc! 安装成功
    goto :eof
) else (
    set /a "retry_count+=1"
    if !retry_count! geq %MAX_RETRIES% (
        echo 错误：!install_desc! 安装失败 - 网络不可达
        pause
        exit /b 1
    )
    echo !install_desc! 安装失败，10秒后重试(!retry_count!/%MAX_RETRIES%)...
    timeout /t 10 /nobreak >nul
    goto retry_loop
)

:: ==================== 创建conda环境 ====================
set "env_created=0"
set "retry_count=0"

:create_env_loop
echo 正在创建conda环境(!CONDA_ENV_NAME!)...
conda create -n !CONDA_ENV_NAME! python=!PYTHON_VERSION! -y
if %errorlevel% equ 0 (
    set "env_created=1"
    echo 环境创建成功
) else (
    set /a "retry_count+=1"
    if !retry_count! geq !MAX_RETRIES! (
        echo 错误：环境创建失败 - 网络不可达或conda错误
        pause
        exit /b 1
    )
    echo 环境创建失败，10秒后重试(!retry_count!/!MAX_RETRIES!)...
    timeout /t 10 /nobreak >nul
    goto create_env_loop
)

:: 激活环境
echo 正在激活conda环境...
call conda.bat activate !CONDA_ENV_NAME!
if %errorlevel% neq 0 (
    echo 错误：无法激活!CONDA_ENV_NAME!环境
    pause
    exit /b 1
)

:: ==================== 安装依赖包 ====================
echo.
echo 开始安装依赖包...

call :install_with_retry "conda install -c conda-forge ffmpeg -y" "FFmpeg"
call :install_with_retry "conda install -c conda-forge pynini==2.1.6 -y" "Pynini"
call :install_with_retry "pip install -r requirements.txt" "基础依赖"
call :install_with_retry "pip install WeTextProcessing==1.0.3" "WeTextProcessing"
call :install_with_retry "pip install -e .[webui]" "WebUI扩展"

:: 安装PyTorch
if "!pytorch_version!"=="cpu" (
    call :install_with_retry "pip install torch torchvision torchaudio" "CPU版PyTorch"
) else (
    call :install_with_retry "pip uninstall torch torchaudio -y" "卸载旧版PyTorch"
    call :install_with_retry "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!pytorch_version!" "CUDA版PyTorch"
)

call :install_with_retry "pip install modelscope" "ModelScope"

:: ==================== 下载模型 ====================
echo.
echo 正在下载IndexTTS模型...
modelscope download --model IndexTeam/IndexTTS-1.5 --local_dir ./checkpoints/
if %errorlevel% neq 0 (
    echo 警告：模型下载失败，请手动下载
    echo 下载地址: https://modelscope.cn/models/IndexTeam/IndexTTS-1.5
)

:: ==================== 完成提示 ====================
echo.
echo =========================================
echo  安装成功！
echo =========================================
echo.
echo 使用方法:
echo 双击run.bat启动
echo.
echo 如果遇到问题，请检查:
echo - 确保CUDA驱动已安装并更新到最新版本
echo - 检查网络连接是否正常
echo - 查看requirements.txt文件是否存在
echo =========================================
pause
exit /b 0
