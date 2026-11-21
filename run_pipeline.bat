@echo off
REM ==================================================================
REM  Conflict Resolution in Multi-Agent Path Finding
REM  via Neural Imitation Learning  -  end-to-end pipeline
REM
REM  Stage 1 : generate expert demonstrations with the CBS solver
REM  Stage 2 : train the neural policy (behavioural cloning)
REM  Stage 3 : benchmark greedy / neural / neural + safety shield
REM ==================================================================
setlocal
set PY=C:\Users\hoora\anaconda3\python.exe
cd /d "%~dp0"

echo.
echo ============================================================
echo   Stage 1 of 3  -  Building the expert dataset (CBS)
echo ============================================================
cd task1
"%PY%" generate_warehouse_dataset.py --n 1500 --out wh_dataset --seed 42
cd ..
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo   Stage 2 of 3  -  Training the neural policy
echo ============================================================
"%PY%" task2\train_v3.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo   Stage 3 of 3  -  Benchmarking the policies
echo ============================================================
echo.
echo   [ Baseline: greedy, no coordination ]
"%PY%" task4\evaluate.py --dataset task1\wh_dataset --policy greedy
echo.
echo   [ Neural policy ]
"%PY%" task4\evaluate.py --dataset task1\wh_dataset --policy model --model model.pt
echo.
echo   [ Neural policy + safety shield ]
"%PY%" task4\evaluate.py --dataset task1\wh_dataset --policy model --model model.pt --shield

echo.
echo ============================================================
echo   Pipeline complete.
echo ============================================================
goto :end

:fail
echo.
echo   A stage did not complete. Review the message above.

:end
pause
endlocal
