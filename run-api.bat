@echo off
chcp 65001 >nul

call conda activate index-tts

python api.py

pause
