@echo off
REM  LIVE DEMO: held-out evaluation, then the visualizer.
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
cd /d "%~dp0"
echo ============================================================
echo   Preparing the held-out test set (maps never used in training)
echo ============================================================
"%PY%" make_holdout.py
cd task4
echo.
echo   --- Greedy baseline ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy greedy
echo   --- Neural policy (CNN) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt
echo   --- Neural + safety shield (expect 0 collisions) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt --shield
echo   --- CBS optimal expert (reference) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy cbs
echo.
echo ============================================================
echo   Launching the live visualizer
echo ============================================================
"%PY%" visualize.py ..\task1\wh_dataset
endlocal
