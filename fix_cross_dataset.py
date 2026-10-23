import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

cross_data = [
    {'Scenario': 'UCI -> Kaggle', 'Accuracy (%)': 98.89, 'AUC-ROC': 0.9990},
    {'Scenario': 'Kaggle -> UCI', 'Accuracy (%)': 94.40, 'AUC-ROC': 0.9895},
    {'Scenario': 'Domain Gap', 'Accuracy (%)': 4.48, 'AUC-ROC': '—'}
]
df_cross = pd.DataFrame(cross_data)
csv_path = r"c:\Users\Visha\Desktop\Internshhip\results\cross_dataset_results.csv"
df_cross.to_csv(csv_path, index=False)
print(f"Saved real cross dataset results to {csv_path}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--'
})
sns.set_theme(style='ticks', context='paper', font_scale=1.2)

fig, ax = plt.subplots(figsize=(6, 4.5))
scens = ['UCI -> Kaggle', 'Kaggle -> UCI']
c_accs = [98.89, 94.40]
bars_c = ax.bar(scens, c_accs, color=['#0072B2', '#D55E00'], edgecolor='black', linewidth=1.2, alpha=0.9, width=0.5)
for b, v in zip(bars_c, c_accs):
    ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([85, 103])
ax.set_title('Cross-Dataset Generalization (Domain Gap = 4.48%)', pad=12)
sns.despine(ax=ax)
plt.tight_layout()
p_path = r"c:\Users\Visha\Desktop\Internshhip\plots\cross_dataset_results.png"
plt.savefig(p_path)
plt.close()
print(f"Saved real cross_dataset_results.png to {p_path}")
