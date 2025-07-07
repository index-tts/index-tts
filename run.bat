@echo off
chcp 65001 >nul

start "" cmd /c "timeout /t 25 >nul && start "" "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" --new-window "http://127.0.0.1:7860/""

cd /d 【安装位置】

call conda activate index-tts

python webui.py

pause