import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv('results_subject_independent/subject_independent_results.csv')

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Bar chart comparing trial-level vs subject-level accuracy per fold
folds = results['fold']
trial_acc = results['test_acc_trial'] * 100
subject_acc = results['test_acc_subject'] * 100

x = np.arange(len(folds))
width = 0.35

bars1 = ax1.bar(x - width/2, trial_acc, width, label='Trial-Level', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, subject_acc, width, label='Subject-Level', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Trial-Level vs Subject-Level Accuracy\nper Cross-Validation Fold', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Fold {i}' for i in folds])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(60, 85)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: Summary statistics with error bars
mean_trial = trial_acc.mean()
std_trial = trial_acc.std()
mean_subject = subject_acc.mean()
std_subject = subject_acc.std()

ax2.bar(['Trial-Level', 'Subject-Level'], [mean_trial, mean_subject],
        yerr=[std_trial, std_subject], capsize=10,
        color=['#3498db', '#e74c3c'], alpha=0.8, width=0.6)
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Subject-Independent Performance Summary\n(3-Fold Nested CV)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(65, 80)

# Add text annotations
ax2.text(0, mean_trial + std_trial + 1, f'{mean_trial:.2f}% ± {std_trial:.2f}%',
         ha='center', fontsize=10, fontweight='bold')
ax2.text(1, mean_subject + std_subject + 1, f'{mean_subject:.2f}% ± {std_subject:.2f}%',
         ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/subject_independent_results.png', dpi=300, bbox_inches='tight')
print("Plot saved to: plots/subject_independent_results.png")
plt.close()
