@echo off
REM  One-click: launch the visualizer only.
setlocal
REM --- locate Anaconda Python (portable: works on any machine/user) ---
set "PY="
if exist "%USERPROFILE%\anaconda3\python.exe"  set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%ProgramData%\anaconda3\python.exe"   set "PY=%ProgramData%\anaconda3\python.exe"
if not defined PY ( python --version >nul 2>&1 && set "PY=python" )
if not defined PY (
  echo [!] Could not find Anaconda Python automatically.
  echo     Open "Anaconda Prompt" and run this from there, or edit this
  echo     file and set PY to the full path of your python.exe
  pause
  exit /b
)
cd /d "%~dp0\task4"
"%PY%" visualize.py ..\task1\wh_dataset
endlocal
