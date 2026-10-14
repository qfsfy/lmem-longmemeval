@echo off
chcp 65001 >nul
title mem0 demo
cd /d "%~dp0"

echo ============================================
echo   mem0 demo launcher
echo   (Chinese output comes from the Python scripts)
echo ============================================
echo   NOTE: mem0 is NOT a resident service. The script
echo   runs once and exits. Data lives in runs\mem0-demo\.
echo   Delete that folder to reset everything.
echo.

if not "%ZAI_API_KEY%"=="" goto have_key

set "KEY="
set /p "KEY=Paste your Zhipu API Key then press Enter (empty = offline demo, no key needed): "

if "%KEY%"=="" goto offline
set "ZAI_API_KEY=%KEY%"
goto real

:have_key
echo Found ZAI_API_KEY in environment variables, using it.
goto real

:offline
echo.
echo ---------- [offline demo, no key] ----------
".venv-mem0\Scripts\python.exe" scripts\mem0_demo_offline.py
goto end

:real
echo.
echo ---------- [real GLM demo] ----------
".venv-mem0\Scripts\python.exe" scripts\mem0_demo.py

:end
echo.
echo ============================================
echo  Done. To reset demo data: delete runs\mem0-demo
echo ============================================
pause
