import os
import re
import numpy as np
import pandas as pd
from scipy import stats

print("="*70)
print("  DEEP SCIENTIFIC & CODE AUDIT")
print("="*70)

# 1. Check biostg_cv_results.csv
df_cv = pd.read_csv('results/biostg_cv_results.csv')
print("\n[1. CV RESULTS AUDIT]")
print(f"Number of folds: {len(df_cv)}")
mean_acc = df_cv['val_acc'].mean() * 100
std_acc = df_cv['val_acc'].std() * 100
mean_auc = df_cv['auc_roc'].mean() * 100
std_auc = df_cv['auc_roc'].std() * 100
mean_f1 = df_cv['f1'].mean()
print(f"Mean Accuracy: {mean_acc:.4f}% ± {std_acc:.4f}%")
print(f"Mean AUC-ROC:  {mean_auc:.4f}% ± {std_auc:.4f}%")
print(f"Mean F1:       {mean_f1:.4f}")
print("Per-fold validation accuracies:")
print(df_cv[['fold', 'val_acc', 'f1', 'auc_roc']])

# 2. Check Predictions and Probabilities consistency
y_true = np.load('results/biostg_true.npy')
y_pred = np.load('results/biostg_pred.npy')
y_prob = np.load('results/biostg_prob.npy')
print("\n[2. PREDICTION ARRAYS AUDIT]")
print(f"y_true shape: {y_true.shape}, sum: {y_true.sum()} (Alcoholic: {y_true.sum()}, Control: {len(y_true)-y_true.sum()})")
print(f"y_pred shape: {y_pred.shape}")
print(f"y_prob shape: {y_prob.shape}, range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
overall_acc = (y_true == y_pred).mean() * 100
errors = (y_true != y_pred).sum()
print(f"Overall Accuracy across 1,200 trials: {overall_acc:.4f}% (Errors: {errors} out of 1,200)")

# 3. Check Baseline Comparison CSV
df_base = pd.read_csv('results/baseline_comparison.csv')
print("\n[3. BASELINES COMPARISON AUDIT]")
print(df_base)

# 4. Check Statistical Significance Tests CSV
df_stat = pd.read_csv('results/statistical_tests.csv')
print("\n[4. STATISTICAL SIGNIFICANCE TESTS AUDIT]")
print(df_stat)

# 5. Check Ablation Study CSV
df_abl = pd.read_csv('results/ablation_study.csv')
print("\n[5. ABLATION STUDY AUDIT]")
print(df_abl)

# 6. Check Cross-Dataset CSV
df_cross = pd.read_csv('results/cross_dataset_results.csv')
print("\n[6. CROSS-DATASET AUDIT]")
print(df_cross)

# 7. Check All Plots Exist and Non-Zero
plots = [
    'full_training_curves.png',
    'full_confusion_matrix.png',
    'full_roc_curve.png',
    'full_fold_accuracy.png',
    'full_comparison.png',
    'full_significance.png',
    'ablation_results.png',
    'cross_dataset_results.png',
    'topomap_attention_weights.png'
]
print("\n[7. PLOTS EXISTENCE & INTEGRITY AUDIT]")
for p in plots:
    path = os.path.join('plots', p)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  [OK] {p:30s} ({size:,} bytes)")
    else:
        print(f"  [MISSING ERROR] {p:30s} NOT FOUND!")

# 8. Check LaTeX files for exact number matching
print("\n[8. LATEX PAPERS CONSISTENCY AUDIT]")
tex_files = ['Final_Paper_EEG_Part1.tex', 'Final_Paper_EEG_Part2.tex', 'Final_Paper_EEG_Part3.tex']
for tf in tex_files:
    content = open(tf, 'r', encoding='utf-8').read()
    # Check for old numbers
    old_numbers = ['96.50', '95.50', '95.58', '93.33', '73.56', '83.00', '83.17']
    found_old = []
    for on in old_numbers:
        if on in content:
            found_old.append(on)
    if found_old:
        print(f"  [WARNING in {tf}] Found potential stale metrics: {found_old}")
    else:
        print(f"  [CLEAN] {tf} contains no stale numbers.")

    # Check for missing plot references
    fig_refs = re.findall(r'\\includegraphics\[.*?\]\{(plots/.*?)\}', content)
    for fr in fig_refs:
        if not os.path.exists(fr):
            print(f"  [BROKEN FIGURE LINK in {tf}] {fr} does not exist!")
        else:
            print(f"  [FIGURE LINK OK] {fr}")

print("\n" + "="*70)
print("  AUDIT COMPLETE")
print("="*70)
