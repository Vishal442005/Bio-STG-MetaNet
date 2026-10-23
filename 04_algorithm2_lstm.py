"""
FILE: 04_algorithm2_lstm.py
PROJECT: Alcoholic EEG Classification - Internship Research
PAPER: Farsi et al. (2021) — Algorithm 2 (PRIMARY)
PURPOSE: Direct LSTM on raw EEG data — NO hand-crafted feature extraction.
         Replicates Algorithm 2 from the paper (93% accuracy on UCI dataset).

ALGORITHM 2 PIPELINE:
  Raw EEG (64ch × 256t) → LSTM network → Dense → Softmax → Class

KEY INSIGHT (from paper):
  Deep learning on raw data > hand-crafted features + DL
  (Algorithm 2 at 93% beats Algorithm 1 at 86%)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Visha\Desktop\Internshhip\data"
OUT_DIR    = r"C:\Users\Visha\Desktop\Internshhip\outputs"

BATCH_SIZE = 32
EPOCHS     = 50
LR         = 5e-4

# LSTM Architecture (mirroring paper config)
INPUT_SIZE  = 64    # Number of EEG channels fed per time-step
HIDDEN_SIZE = 128   # LSTM hidden units
NUM_LAYERS  = 2     # Stacked LSTM layers
N_CLASSES   = 2     # Alcoholic / Control
DROPOUT     = 0.3
BIDIRECT    = True  # Bidirectional LSTM (our improvement over paper)

RANDOM_STATE = 42
# ────────────────────────────────────────────────────────────


class LSTMClassifier(nn.Module):
    """
    LSTM-based EEG classifier (Algorithm 2 — Farsi et al. 2021).

    Input:  (Batch, Channels=64, TimeSteps=256)
    Flow:
        Permute → (Batch, TimeSteps, Channels)  [LSTM expects seq-first]
        LSTM    → (Batch, TimeSteps, Hidden*2)
        Take last timestep output
        Dense   → (Batch, N_Classes)
    """
    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        n_classes=N_CLASSES,
        dropout=DROPOUT,
        bidirectional=BIDIRECT
    ):
        super().__init__()
        self.bidirectional = bidirectional
        fc_in = hidden_size * 2 if bidirectional else hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.layer_norm = nn.LayerNorm(fc_in)
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (B, C=64, T=256)
        x = x.permute(0, 2, 1)       # → (B, T, C)
        out, _ = self.lstm(x)         # → (B, T, H*2)
        last = out[:, -1, :]          # last time-step
        last = self.layer_norm(last)
        return self.classifier(last)


def run_training(model, train_ld, val_ld, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_acc, best_weights = 0.0, None

    for ep in range(1, epochs + 1):
        # Train
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optimizer.step()
            t_loss += loss.item() * xb.size(0)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total += xb.size(0)

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * xb.size(0)
                preds = logits.argmax(1)
                v_correct += (preds == yb).sum().item()
                v_total += yb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        tr_acc = t_correct / t_total * 100
        vl_acc = v_correct / v_total * 100
        scheduler.step(v_loss / v_total)

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['train_loss'].append(t_loss / t_total)
        history['val_loss'].append(v_loss / v_total)

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0:
            print(f"  Ep {ep:2d}/{epochs} | Train: {tr_acc:.2f}% | Val: {vl_acc:.2f}%")

    if best_weights:
        model.load_state_dict(best_weights)

    return history, best_acc, all_preds, all_labels


def plot_all_results(history, all_preds, all_labels, best_acc, save_dir):
    """Comprehensive results visualization."""
    fig = plt.figure(figsize=(18, 10), facecolor='#0d1526')
    fig.suptitle(
        f'Algorithm 2: LSTM on Raw EEG Data — Best Val Acc: {best_acc:.2f}%',
        color='white', fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def style_ax(ax):
        ax.set_facecolor('#060b18')
        ax.tick_params(colors='#5d7499')
        ax.spines[:].set_color('#1e2d45')

    epochs_range = range(1, len(history['train_acc']) + 1)

    # Accuracy
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)
    ax1.plot(epochs_range, history['train_acc'], '#00f0ff', lw=2, label='Training Acc')
    ax1.plot(epochs_range, history['val_acc'],   '#10d9a0', lw=2, label='Validation Acc')
    ax1.axhline(93, color='#f59e0b', ls='--', alpha=0.8, label='Paper Target (93%)')
    ax1.axhline(95, color='#8b5cf6', ls='--', alpha=0.8, label='Our Target (95%)')
    ax1.set_title('Accuracy Over Training', color='white')
    ax1.set_xlabel('Epoch', color='#94a8c4')
    ax1.set_ylabel('Accuracy (%)', color='#94a8c4')
    ax1.legend(facecolor='#0d1526', labelcolor='white')
    ax1.set_ylim([40, 100])

    # Loss
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)
    ax2.plot(epochs_range, history['train_loss'], '#ef4444', lw=2, label='Train')
    ax2.plot(epochs_range, history['val_loss'],   '#f59e0b', lw=2, label='Val')
    ax2.set_title('Loss Curves', color='white')
    ax2.set_xlabel('Epoch', color='#94a8c4')
    ax2.set_ylabel('CrossEntropy Loss', color='#94a8c4')
    ax2.legend(facecolor='#0d1526', labelcolor='white')

    # Confusion matrix
    ax3 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax3, cmap='YlOrRd',
                xticklabels=['Control', 'Alcoholic'],
                yticklabels=['Control', 'Alcoholic'])
    ax3.set_title('Confusion Matrix', color='white')
    ax3.set_facecolor('#060b18')
    ax3.tick_params(colors='white')

    # Per-class accuracy bar
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    class_accs = [cm[i, i] / cm[i].sum() * 100 for i in range(2)]
    bars = ax4.bar(['Control', 'Alcoholic'], class_accs,
                   color=['#10d9a0', '#ef4444'], width=0.5, edgecolor='white', linewidth=0.5)
    for bar, acc in zip(bars, class_accs):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', color='white', fontsize=11, fontweight='bold')
    ax4.set_title('Per-Class Accuracy', color='white')
    ax4.set_ylabel('Accuracy (%)', color='#94a8c4')
    ax4.set_ylim([0, 110])

    # Algorithm comparison bar
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5)
    methods  = ['Alg 1\n(PCA+ANN)', 'Alg 2\n(LSTM)', 'Our\nLSTM', 'Target']
    accs     = [86, 93, best_acc, 95]
    colors   = ['#5d7499', '#f59e0b', '#00f0ff', '#8b5cf6']
    bars2 = ax5.bar(methods, accs, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, acc in zip(bars2, accs):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', color='white', fontsize=10, fontweight='bold')
    ax5.set_title('Algorithm Comparison', color='white')
    ax5.set_ylabel('Accuracy (%)', color='#94a8c4')
    ax5.set_ylim([70, 100])

    plt.tight_layout()
    path = str(Path(save_dir) / 'alg2_full_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved → {path}")
    plt.show()
    plt.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("  Algorithm 2: LSTM on Raw EEG (Farsi et al. 2021)")
    print("=" * 60)

    data_path = Path(DATA_DIR)
    out_path  = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    X = np.load(data_path / 'X_filtered_fir.npy')  # use FIR-filtered data
    y = np.load(data_path / 'y_train.npy')
    print(f"Data: {X.shape}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    train_ld = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_ld = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=BATCH_SIZE, num_workers=0
    )

    model = LSTMClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    print("\nTraining...")
    history, best_acc, final_preds, final_labels = run_training(
        model, train_ld, val_ld, device
    )

    torch.save(model.state_dict(), str(out_path / 'alg2_best_lstm.pth'))
    print(f"\n>>> Best Validation Accuracy (Algorithm 2 / LSTM): {best_acc:.2f}%")
    print(f">>> Paper benchmark: 93.00%")
    print(f">>> Delta: {best_acc - 93:.2f}%")

    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=['Control', 'Alcoholic']))

    plot_all_results(history, final_preds, final_labels, best_acc, save_dir=str(out_path))
