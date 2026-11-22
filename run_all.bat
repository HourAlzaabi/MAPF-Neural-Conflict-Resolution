@echo off
REM ============================================================
REM  MAPF - Conflict Resolution via Neural Imitation Learning
REM  One entry point for everything. Just double-click this file.
REM ============================================================
setlocal enabledelayedexpansion
set "PY=C:\Users\hoora\anaconda3\python.exe"
cd /d "%~dp0"

:menu
echo(
echo ============================================================
echo   MAPF  -  Conflict Resolution via Neural Imitation Learning
echo ============================================================
echo(
echo   [1]  Full pipeline     rebuild dataset - retrain - evaluate   (slow)
echo   [2]  Evaluate only      honest held-out numbers
echo   [3]  Visualizer only    watch the robots
echo   [4]  Demo               held-out evaluation, then the visualizer
echo   [5]  Shield tests       run the safety unit tests
echo   [Q]  Quit
echo(
set /p "choice=Choose an option (1-5 or Q): "
if /i "!choice!"=="1" ( call :pipeline & goto menu )
if /i "!choice!"=="2" ( call :evaluate & goto menu )
if /i "!choice!"=="3" ( call :visual   & goto menu )
if /i "!choice!"=="4" ( call :demo      & goto menu )
if /i "!choice!"=="5" ( call :tests     & goto menu )
if /i "!choice!"=="Q" goto :eof
goto menu

:pipeline
echo.
echo === 1/3  Rebuilding the dataset (seed 42) ===
pushd task1
"%PY%" generate_warehouse_dataset.py --n 1500 --out wh_dataset --seed 42
popd
echo.
echo === 2/3  Training the neural policy ===
"%PY%" task2\train_v3.py
echo.
echo === 3/3  Evaluating on held-out maps ===
call :evaluate
exit /b

:evaluate
"%PY%" make_holdout.py
pushd task4
echo.
echo   --- Greedy baseline ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy greedy
echo   --- Neural policy (CNN) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt
echo   --- Neural + safety shield (expect 0 collisions) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt --shield
echo   --- CBS optimal expert (reference) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy cbs
popd
echo.
pause
exit /b

:visual
pushd task4
"%PY%" visualize.py ..\task1\wh_dataset
popd
exit /b

:demo
"%PY%" make_holdout.py
pushd task4
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy greedy
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt --shield
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy cbs
echo.
echo Launching the live visualizer...
"%PY%" visualize.py ..\task1\wh_dataset
popd
exit /b

:tests
"%PY%" -m pytest task3 -q
echo.
pause
exit /b
