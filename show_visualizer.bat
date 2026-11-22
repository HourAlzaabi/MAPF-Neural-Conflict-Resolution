@echo off
REM One-click: launch the live warehouse visualizer (no evaluation).
REM N / P = next / previous scenario,  SPACE = play/pause,  T = theme,  Q = quit
setlocal
set PY=C:\Users\hoora\anaconda3\python.exe
cd /d "%~dp0\task4"
"%PY%" visualize.py ..\task1\wh_dataset
endlocal
