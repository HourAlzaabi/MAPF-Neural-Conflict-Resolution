@echo off
REM ============================================================
REM  Conflict Resolution in Multi-Agent Path Finding
REM  via Neural Imitation Learning  -  LIVE DEMO
REM
REM  Runs the honest evaluation on held-out (unseen) maps,
REM  then launches the live visualizer.
REM ============================================================
setlocal
set PY=C:\Users\hoora\anaconda3\python.exe
cd /d "%~dp0"

echo.
echo ============================================================
echo   Preparing the held-out test set (maps never used in training)
echo ============================================================
"%PY%" make_holdout.py

cd task4
echo.
echo ============================================================
echo   Evaluation on held-out maps  (success / collisions / cost)
echo ============================================================
echo.
echo   --- Baseline: greedy, no coordination ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy greedy
echo.
echo   --- Neural policy (our CNN) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt
echo.
echo   --- Neural policy + safety shield  (expect 0 collisions) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt --shield
echo.
echo   --- CBS optimal expert (reference) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy cbs

echo.
echo ============================================================
echo   Launching the live visualizer
echo   (N / P = next / previous scenario,  SPACE = play,  Q = quit)
echo ============================================================
"%PY%" visualize.py ..\task1\wh_dataset

endlocal
