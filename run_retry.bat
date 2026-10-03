@echo off
REM Re-run ONLY the pipeline (deps installed + data already prepared).
cd /d "%~dp0"
del /q run_done.flag 2>nul
echo === running pipeline (retry) === > run_log2.txt
python qedhtn_pipeline.py --config run.example.json --outdir all_tables --max-cardinality 50 >> run_log2.txt 2>&1
echo DONE > run_done.flag
echo ===== FINISHED ===== >> run_log2.txt
