@echo off
title Sign Language - Gesture Data Manager
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\python.exe" (
    set "PY_EXE=..\venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

%PY_EXE% backend\app\manage_data.py
pause
