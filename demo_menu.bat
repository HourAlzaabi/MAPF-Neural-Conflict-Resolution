@echo off
REM ============================================================
REM   MAPF  -  VISUAL DEMO MENU
REM   Pick what to visualize: CBS, CNN alone, CNN + Shield,
REM   or the "shield saves the day" side-by-side on the SAME maps.
REM   Each scenario is labelled live:  SUCCESS / COLLISION / DEADLOCK.
REM ============================================================
setlocal enabledelayedexpansion

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

REM --- make sure the held-out set and curated demo sets exist ---
if not exist "task1\wh_holdout" (
  if not exist "task1\wh_dataset" (
    echo [!] No dataset found. Run run_pipeline.bat once first ^(builds + trains^).
    pause & exit /b
  )
  echo Preparing the held-out test set ...
  "%PY%" make_holdout.py
)
if not exist "demo_sets" (
  echo Building curated demo sets ^(first run only, ~20s^) ...
  "%PY%" make_demo_sets.py
)

:menu
echo(
echo ============================================================
echo    MAPF  -  what do you want to VISUALIZE?
echo ============================================================
echo(
echo    [1]  CBS             the optimal expert  ^(always solves^)
echo    [2]  CNN alone       a map it SOLVES     ^(no shield, still works^)
echo    [3]  CNN alone       watch it CRASH      ^(no shield^)
echo    [4]  CNN + Shield    zero collisions
echo    [5]  Shield saves the day   SAME maps: CNN crashes, then CNN+Shield fixes it
echo    [6]  CNN + Shield    a DEADLOCK case     ^(the honest limitation^)
echo    [7]  Greedy          no coordination     ^(lots of crashes^)
echo(
echo    [8]  Print the NUMBERS table  ^(held-out evaluation, all 4 policies^)
echo    [9]  Rebuild the curated demo sets
echo    [Q]  Quit
echo(
echo    In the window:  SPACE play/pause  -  N next map  -  R restart  -  Q close
echo(
set /p "choice=Choose (1-9 or Q): "

if /i "!choice!"=="1" ( call :vis "1_cbs"        ""                                  "CBS - the optimal expert" & goto menu )
if /i "!choice!"=="2" ( call :vis "3_cnn_ok"     "--policy model --model model.pt"   "CNN ALONE - a map it solves, no shield" & goto menu )
if /i "!choice!"=="3" ( call :vis "2_cnn_crash"  "--policy model --model model.pt"   "CNN ALONE - expect collisions" & goto menu )
if /i "!choice!"=="4" ( call :vis "5_shield_ok"  "--policy model --model model.pt --shield" "CNN + SHIELD - zero collisions" & goto menu )
if /i "!choice!"=="5" ( call :saves & goto menu )
if /i "!choice!"=="6" ( call :vis "6_shield_deadlock" "--policy model --model model.pt --shield" "CNN + SHIELD - a deadlock case" & goto menu )
if /i "!choice!"=="7" ( call :vis "2_cnn_crash"  "--policy greedy"                   "GREEDY - no coordination" & goto menu )
if /i "!choice!"=="8" ( call :numbers & goto menu )
if /i "!choice!"=="9" ( "%PY%" make_demo_sets.py & goto menu )
if /i "!choice!"=="Q" goto :eof
goto menu

:vis
REM  %~1 = demo_sets subfolder   %~2 = visualize flags   %~3 = caption
echo(
echo --- %~3 ---
"%PY%" task4\visualize.py "demo_sets\%~1" %~2
exit /b

:saves
echo(
echo === Shield saves the day ===
echo These are the SAME maps. First you'll see the CNN with NO shield - it collides.
echo Close the window ^(Q^), and it reopens the SAME maps WITH the shield - zero collisions.
echo(
pause
echo --- 1/2 : CNN, NO shield  ^(watch the red COLLISION^) ---
"%PY%" task4\visualize.py "demo_sets\4_shield_saves" --policy model --model model.pt
echo --- 2/2 : CNN + SHIELD, same maps  ^(now SUCCESS, 0 collisions^) ---
"%PY%" task4\visualize.py "demo_sets\4_shield_saves" --policy model --model model.pt --shield
exit /b

:numbers
echo(
"%PY%" make_holdout.py
pushd task4
echo   --- Greedy baseline ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy greedy
echo   --- Neural policy ^(CNN^) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt
echo   --- Neural + safety shield ^(expect 0 collisions^) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy model --model ..\model.pt --shield
echo   --- CBS optimal expert ^(reference^) ---
"%PY%" evaluate.py --dataset ..\task1\wh_holdout --policy cbs
popd
echo(
pause
exit /b