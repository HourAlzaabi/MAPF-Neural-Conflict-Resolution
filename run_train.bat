@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set LOG=_run_log.txt
echo === run start %DATE% %TIME% === > "%LOG%"
call "%UserProfile%\anaconda3\Scripts\activate.bat" >> "%LOG%" 2>&1
echo --- searching envs for torch --- >> "%LOG%"
set FOUND=
for /f "tokens=1" %%e in ('conda env list ^| findstr /v "#"') do (
  if not defined FOUND (
    call conda activate %%e >nul 2>&1
    python -c "import torch" >nul 2>&1
    if !errorlevel!==0 (
      set FOUND=%%e
      echo torch found in env: %%e >> "%LOG%"
    )
  )
)
if not defined FOUND (
  echo NO_TORCH_ENV >> "%LOG%"
  conda env list >> "%LOG%" 2>&1
  echo ===DONE=== >> "%LOG%"
  exit /b
)
call conda activate !FOUND!
python -c "import torch,numpy;print('TORCH_OK',torch.__version__)" >> "%LOG%" 2>&1
echo --- train (task2/train_v3.py) --- >> "%LOG%"
python task2\train_v3.py >> "%LOG%" 2>&1
echo --- evaluate model --- >> "%LOG%"
python task4\evaluate.py --dataset task1\wh_dataset --policy model --model model.pt >> "%LOG%" 2>&1
echo ===DONE=== >> "%LOG%"
