@echo off
chcp 65001 >nul
title CS-Agent 智能体
cd /d "%~dp0"

set PY=..\.venv\Scripts\pythonw.exe
if not exist "%PY%" set PY=pythonw.exe

start "" "%PY%" desktop_app.py
