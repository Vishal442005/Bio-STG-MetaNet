@echo off
REM ============================================================
REM UNIFIED BATCH FILE TO RUN EEG CLASSIFICATION MASTER PIPELINE
REM ============================================================
REM This script runs the entire pipeline, addressing all reviewer
REM concerns (cross-dataset, ablation, baseline comparison, 
REM latency, and >97% accuracy goal).
REM 
REM It will automatically update Final_Paper_EEG.tex at the end!
REM ============================================================

echo ============================================================
echo 🧠 COMPREHENSIVE EEG CLASSIFICATION EXPERIMENTS (MASTER)
echo ============================================================
echo.

REM 1. Ensure virtual environment is activated
if exist .venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] No .venv found in this directory. Using global Python.
)

REM 2. Ensure all required packages are installed
echo.
echo [INFO] Checking Python dependencies...
pip install seaborn

REM 3. Create necessary output directories
if not exist plots mkdir plots
if not exist results mkdir results
if not exist models mkdir models

echo.
echo ============================================================
echo [STEP 1] Running Master Pipeline...
echo ============================================================
echo This will execute:
echo   - Task 1: Pilot training on small dataset for >97%% accuracy
echo   - Ablation study across 8 configurations
echo   - Baseline model comparisons (LSTM, CNN, PCA+ANN)
echo   - Cross-dataset validation
echo   - Task 2: Full dataset training
echo.
echo Note: This process will take 30-120 minutes depending on CPU/GPU.
echo Do not close this window.
echo ------------------------------------------------------------
python eeg_master_pipeline.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Master pipeline failed during execution.
    pause
    exit /b 1
)
echo [SUCCESS] Master pipeline completed successfully!
echo.

echo ============================================================
echo [STEP 2] Updating LaTeX Paper...
echo ============================================================
python update_latex.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] LaTeX update failed.
    pause
    exit /b 1
)
echo [SUCCESS] LaTeX paper updated successfully!
echo.

echo ============================================================
echo 🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo Check the following directories for your outputs:
echo   - plots/       (Training curves, bar charts, radar charts, ROC)
echo   - results/     (CSV tables and all_latex_tables.tex)
echo   - models/      (Saved .pth weights)
echo.
echo Your updated paper is saved as: Final_Paper_EEG_UPDATED.tex
echo ============================================================
pause