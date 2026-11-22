@echo off
REM  Full pipeline: build dataset -> train policy -> benchmark (held-out).
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
echo === 1/3  Building the dataset (seed 42) ===
pushd task1 & "%PY%" generate_warehouse_dataset.py --n 1500 --out wh_dataset --seed 42 & popd
echo === 2/3  Training the neural policy ===
"%PY%" task2\train_v3.py
echo === 3/3  Evaluating on held-out maps ===
"%PY%" make_holdout.py
cd task4
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy greedy
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt --shield
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy cbs
echo Done.
pause
endlocal
