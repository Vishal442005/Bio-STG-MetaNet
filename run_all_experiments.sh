#!/bin/bash
# ============================================================
# UNIFIED BASH FILE TO RUN EEG CLASSIFICATION MASTER PIPELINE
# ============================================================

echo "============================================================"
echo "🧠 COMPREHENSIVE EEG CLASSIFICATION EXPERIMENTS (MASTER)"
echo "============================================================"
echo ""

# 1. Ensure all required packages are installed (using user installation)
echo "[INFO] Checking Python dependencies..."
python3 -m pip install torch numpy scipy scikit-learn pandas matplotlib seaborn -q --user
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi

# 2. Create necessary output directories
mkdir -p plots results models

echo ""
echo "============================================================"
echo "[STEP 1] Running Master Pipeline..."
echo "============================================================"
python3 eeg_master_pipeline.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Master pipeline failed during execution."
    exit 1
fi
echo "[SUCCESS] Master pipeline completed successfully!"
echo ""

echo "============================================================"
echo "[STEP 2] Updating LaTeX Paper..."
echo "============================================================"
python3 update_latex.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] LaTeX update failed."
    exit 1
fi
echo "[SUCCESS] LaTeX paper updated successfully!"
echo ""

echo "============================================================"
echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "============================================================"
