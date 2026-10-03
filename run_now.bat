@echo off
REM Runs the full pipeline on already-downloaded data. Logs everything to files.
cd /d "%~dp0"
del /q run_done.flag 2>nul
echo === installing dependencies === > run_log.txt
python -m pip install --quiet --upgrade pip >> run_log.txt 2>&1
python -m pip install --quiet pandas scikit-learn numpy scipy >> run_log.txt 2>&1
echo === preparing datasets === >> run_log.txt
python prepare_local.py >> run_log.txt 2>&1
echo === running pipeline === >> run_log.txt
python qedhtn_pipeline.py --config run.example.json --outdir all_tables --max-cardinality 50 >> run_log.txt 2>&1
echo DONE > run_done.flag
echo. >> run_log.txt
echo ===== FINISHED ===== >> run_log.txt
