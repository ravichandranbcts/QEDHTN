@echo off
REM ============================================================
REM  QEDHTN - one-shot download + run (Windows)
REM  Double-click this file, or run it from a terminal, inside
REM  the folder that contains download_datasets.py, qedhtn_*.py
REM  and run.example.json.
REM
REM  ONE-TIME PREREQUISITES (cannot be scripted - they are tied
REM  to your Kaggle account):
REM    1) Create a Kaggle API token (kaggle.com -> Settings ->
REM       Create New Token) and save kaggle.json to
REM       %USERPROFILE%\.kaggle\kaggle.json
REM    2) Accept the IEEE-CIS competition rules once:
REM       https://www.kaggle.com/competitions/ieee-fraud-detection/rules
REM ============================================================

cd /d "%~dp0"
echo.
echo === Step 1/3: installing Python dependencies ===
python -m pip install --upgrade pip
python -m pip install kagglehub pandas scikit-learn numpy scipy
if errorlevel 1 goto :err

echo.
echo === Step 2/3: downloading datasets (PaySim, IEEE-CIS, UCI/ULB) ===
python download_datasets.py
if errorlevel 1 goto :err

echo.
echo === Step 3/3: running the pipeline (5 seeds per stage) ===
python qedhtn_pipeline.py --config run.example.json --outdir all_tables --max-cardinality 50
if errorlevel 1 goto :err

echo.
echo === DONE. Result tables are in the all_tables\ folder. ===
echo Open all_tables\all_tables.txt to view them.
pause
exit /b 0

:err
echo.
echo *** Something failed. Most common causes: ***
echo   - kaggle.json not placed at %USERPROFILE%\.kaggle\kaggle.json
echo   - IEEE-CIS competition rules not accepted yet (see link at top)
echo   - no internet / Kaggle blocked on this network
echo Fix the item above and run this file again.
pause
exit /b 1
