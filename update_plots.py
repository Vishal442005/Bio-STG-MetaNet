import os

file_path = r"c:\Users\Visha\Desktop\Internshhip\complete_code_eeg.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "8.  PLOTTING UTILITIES" in line:
        start_idx = i - 1  # The line with =====
    elif "9.  QUICK TRAIN HELPER" in line:
        end_idx = i - 1  # The line with =====
        break

if start_idx != -1 and end_idx != -1:
    print(f"Found plotting section from line {start_idx} to {end_idx}")
    
    new_code = """# ==============================================================================
# 8.  PLOTTING UTILITIES (PUBLICATION QUALITY)
# ==============================================================================

# Configure publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

sns.set_theme(style='ticks', context='paper', font_scale=1.2)
# Professional colorblind-friendly palette
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']


def plot_training_curves(histories, tag='full', n_folds=10):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    max_epochs = max(len(h['train_acc']) for h in histories)
    mean_train_acc = []
    mean_val_acc = []
    mean_train_loss = []
    mean_val_loss = []
    
    for epoch in range(max_epochs):
        train_accs = [h['train_acc'][epoch] for h in histories if epoch < len(h['train_acc'])]
        val_accs = [h['val_acc'][epoch] for h in histories if epoch < len(h['val_acc'])]
        train_losses = [h['train_loss'][epoch] for h in histories if epoch < len(h['train_loss'])]
        val_losses = [h['val_loss'][epoch] for h in histories if epoch < len(h['val_loss'])]
        
        mean_train_acc.append(np.mean(train_accs) * 100 if train_accs else 0)
        mean_val_acc.append(np.mean(val_accs) * 100 if val_accs else 0)
        mean_train_loss.append(np.mean(train_losses) if train_losses else 0)
        mean_val_loss.append(np.mean(val_losses) if val_losses else 0)
    
    epochs = range(1, max_epochs + 1)
    
    for i, hist in enumerate(histories):
        fold_epochs = range(1, len(hist['train_acc']) + 1)
        axes[0].plot(fold_epochs, [v*100 for v in hist['train_acc']],
                     alpha=0.15, lw=1, color='#0072B2')
        axes[0].plot(fold_epochs, [v*100 for v in hist['val_acc']],
                     alpha=0.15, lw=1, color='#D55E00')
        axes[1].plot(fold_epochs, hist['train_loss'],
                     alpha=0.15, lw=1, color='#0072B2')
        axes[1].plot(fold_epochs, hist['val_loss'],
                     alpha=0.15, lw=1, color='#D55E00')
    
    axes[0].plot(epochs, mean_train_acc, alpha=1.0, lw=2.5, color='#0072B2', 
                 linestyle='-', label='Mean Training Acc')
    axes[0].plot(epochs, mean_val_acc, alpha=1.0, lw=2.5, color='#D55E00', 
                 linestyle='--', label='Mean Validation Acc')
    axes[1].plot(epochs, mean_train_loss, alpha=1.0, lw=2.5, color='#0072B2', 
                 linestyle='-', label='Mean Training Loss')
    axes[1].plot(epochs, mean_val_loss, alpha=1.0, lw=2.5, color='#D55E00', 
                 linestyle='--', label='Mean Validation Loss')

    axes[0].set_title('Accuracy per Epoch', pad=10)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim([40, 102])
    axes[0].legend(loc='lower right', frameon=True, edgecolor='black')
    
    axes[1].set_title('Loss per Epoch', pad=10)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right', frameon=True, edgecolor='black')
    
    sns.despine(fig)
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_training_curves.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")


def plot_confusion_matrix(y_true, y_pred, tag='full'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = f'{cm[i, j]}\\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Control', 'Alcoholic'],
                yticklabels=['Control', 'Alcoholic'],
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=1, linecolor='black', cbar_kws={'label': 'Count'})
    
    ax.set_title(f'Confusion Matrix ({tag})', pad=15)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    ax.text(0.5, -0.2, f'Overall Accuracy: {accuracy:.2f}%', 
            ha='center', va='center', transform=ax.transAxes, 
            fontsize=13, fontweight='bold', color='#0072B2')
    
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_confusion_matrix.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")


def plot_roc_curve(y_true, y_prob, tag='full'):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    
    ax.plot(fpr, tpr, color='#D55E00', lw=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#D55E00')
    
    ax.set_title(f'Receiver Operating Characteristic', pad=15)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    sns.despine(ax=ax)
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_roc_curve.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")


def plot_fold_accuracy(fold_df, tag='full'):
    fig, ax = plt.subplots(figsize=(8, 5))
    accs = fold_df['val_acc'].values * 100
    folds = fold_df['fold'].values
    bars = ax.bar(folds, accs, color=COLORS[:len(folds)], edgecolor='black',
                  linewidth=1.2, alpha=0.9)
    
    mean_acc = accs.mean()
    std_acc = accs.std()
    ax.axhline(mean_acc, color='#D55E00', linestyle='--', lw=2.5,
               label=f'Mean: {mean_acc:.2f}% ± {std_acc:.2f}%')
    
    ax.axhline(97.0, color='#009E73', linestyle=':', lw=2, label='97% Target')
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title('Per-Fold Validation Accuracy', pad=15)
    ax.set_xlabel('Fold Number')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim([max(60, accs.min() - 10), 102])
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    
    sns.despine(ax=ax)
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_fold_accuracy.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")


def plot_ablation(ablation_df, tag='ablation'):
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ablation_df['Configuration'].tolist()
    accs = ablation_df['Mean Acc (%)'].tolist()
    stds = ablation_df['Std Acc (%)'].tolist() if 'Std Acc (%)' in ablation_df.columns else [0] * len(accs)
    
    # Highlight full model
    colors = ['#0072B2' if 'Full' in c else '#56B4E9' for c in configs]
    y_pos = np.arange(len(configs))
    
    bars = ax.barh(y_pos, accs, xerr=stds, align='center', color=colors, 
                   edgecolor='black', linewidth=1.2, alpha=0.9, capsize=5)
    
    for i, (v, std) in enumerate(zip(accs, stds)):
        label = f'{v:.2f}%' if std == 0 else f'{v:.2f} ± {std:.2f}%'
        ax.text(v + std + 1, i, label, va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs, fontsize=12)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Validation Accuracy (%)')
    ax.set_title('Ablation Study: Component Contributions', pad=15)
    ax.set_xlim([max(0, min(accs) - max(stds) - 10), max(accs) + max(stds) + 15])
    
    sns.despine(ax=ax)
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_results.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")


def plot_baseline_comparison(comparison_df, tag='baseline'):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = comparison_df['Model'].tolist()
    accs  = comparison_df['Mean Acc (%)'].tolist()
    aucs  = comparison_df['Mean AUC (%)'].tolist()

    aucs_numeric = [float(a) if (not isinstance(a, str) or a != '—') and a != '—' else 0 for a in aucs]

    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, accs, w, label='Accuracy (%)', color='#0072B2',
                edgecolor='black', linewidth=1.2, alpha=0.9)
    b2 = ax.bar(x + w/2, aucs_numeric, w, label='AUC-ROC (%)', color='#009E73',
                edgecolor='black', linewidth=1.2, alpha=0.9)
    
    for bar, v in zip(b1, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)
    for bar, v in zip(b2, aucs_numeric):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=12)
    ax.set_ylabel('Performance (%)')
    ax.set_title('Comparison with Baseline Models', pad=15)
    ax.set_ylim([min(accs) - 15, 115])
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    
    sns.despine(ax=ax)
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_comparison.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")


def plot_statistical_significance(stat_df, tag='stats'):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r.replace('Proposed vs ', '') for r in stat_df['Comparison']]
    pvals = stat_df['p-value'].values
    cohenD = stat_df["Cohen's d"].values.astype(float)
    x = np.arange(len(names))
    
    # -log10 scaling for better visualization of tiny p-values
    log_pvals = -np.log10(np.abs(pvals) + 1e-10)
    
    colors = ['#D55E00' if p < 0.05 else '#56B4E9' for p in pvals]
    bars = ax.bar(x, log_pvals, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)
    
    sig_line = -np.log10(0.05)
    ax.axhline(sig_line, color='black', linestyle='--', lw=2,
               label='p = 0.05 (Significance Threshold)')
    
    for i, (bar, d, p_val) in enumerate(zip(bars, cohenD, pvals)):
        effect_size = 'Large' if abs(d) > 0.8 else 'Medium' if abs(d) > 0.5 else 'Small'
        sig_text = '*' if p_val < 0.05 else 'n.s.'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"d = {d:.2f}\\n({effect_size})\\n{sig_text}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=12)
    ax.set_ylabel('$-\log_{10}(p\text{-value})$')
    ax.set_title("Statistical Significance (Proposed vs Baselines)", pad=15)
    ax.set_ylim([0, max(log_pvals) + 1.5])
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    
    sns.despine(ax=ax)
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_significance.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")


def plot_per_metric_radar(model_metrics_dict, tag='radar'):
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    for i, (name, vals) in enumerate(model_metrics_dict.items()):
        data = [vals['acc'], vals['prec'], vals['rec'],
                vals['f1'], vals['auc']]
        data += data[:1]
        ax.plot(angles, data, lw=2.5, label=name, color=COLORS[i % len(COLORS)], marker='o', markersize=6)
        ax.fill(angles, data, alpha=0.1, color=COLORS[i % len(COLORS)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
    ax.set_ylim([0.4, 1.05])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'], fontsize=10, color='gray')
    ax.set_title('Multi-Metric Performance Comparison', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), frameon=True, edgecolor='black')
    
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_radar.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")


def plot_cross_dataset(results_dict, tag='cross_dataset'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scenarios = list(results_dict.keys())
    accs = [results_dict[s]['acc']*100 for s in scenarios]
    aucs = [results_dict[s]['auc']*100 for s in scenarios]

    colors_acc = ['#0072B2', '#D55E00']
    colors_auc = ['#009E73', '#CC79A7']
    
    bars_acc = axes[0].bar(scenarios, accs, color=colors_acc,
                edgecolor='black', linewidth=1.2, alpha=0.9)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 1, f'{v:.2f}%', ha='center', fontsize=12, fontweight='bold')
    axes[0].set_ylim([max(50, min(accs)-15), 105])
    axes[0].set_title('Cross-Dataset Accuracy', pad=10)
    axes[0].set_ylabel('Accuracy (%)')
    
    bars_auc = axes[1].bar(scenarios, aucs, color=colors_auc,
                edgecolor='black', linewidth=1.2, alpha=0.9)
    for i, v in enumerate(aucs):
        axes[1].text(i, v + 1, f'{v:.2f}%', ha='center', fontsize=12, fontweight='bold')
    axes[1].set_ylim([max(50, min(aucs)-15), 105])
    axes[1].set_title('Cross-Dataset AUC-ROC', pad=10)
    axes[1].set_ylabel('AUC-ROC (%)')

    sns.despine(fig)
    plt.suptitle('Cross-Dataset Generalisation Performance', y=1.05)
    plt.tight_layout()
    p = os.path.join(CFG['plots_dir'], f'{tag}_results.png')
    plt.savefig(p)
    plt.close()
    print(f"[PLOT] {p}")
"""
    
    lines[start_idx:end_idx+1] = [new_code + "\n"]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Successfully updated the plotting functions.")
else:
    print("Could not find the start or end markers for the plotting section.")
