"""
=============================================================================
  ABLATION CASE 3: LAYER 2 → LAYER 3 (No Layer 1)
  ============================================================================
  Architecture : Layer 2 (Marginal Segregated CNN) → 128-dim
                 → Layer 3 (Meta-Classifier) → 2-class output
  Purpose      : Isolate the contribution of frequency-band segregation
                 (Layer 2) WITHOUT spatio-temporal recurrent modeling (Layer 1).
                 Answers: "Does the multi-scale frequency pathway alone suffice?"
  Saves        : best_model_L23.pth, Train_Val_L23.png
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ─────────────────────────────────────────────────────────────────
#  LAYER 2: Marginal Segregated CNN — Multi-Scale Frequency Pathway
#  Input : (B, 64, 256)
#  Output: (B, 128)
# ─────────────────────────────────────────────────────────────────
class Layer2_MarginalSegregatedCNN(nn.Module):
    """
    Layer 2 — Frequency-Segregated CNN.
    Two parallel convolutional streams process the same EEG signal with
    different kernel sizes, implicitly targeting different EEG frequency bands:

      Slow-Wave Stream (kernel=15, stride=2):
        Large receptive field (≈58ms) → sensitive to δ/θ oscillations (0.5–8 Hz)
        Heavy MaxPool (k=4) further smooths to reveal slow trend patterns.

      Fast-Wave Stream (kernel=3, stride=1):
        Small receptive field (≈12ms) → sensitive to β/γ transients (13–100 Hz)
        MaxPool (k=8) downsamples to match the slow stream's output length.

    The two streams are concatenated and projected to 128-dim via FC layer.
    This design achieves frequency band separation WITHOUT any explicit DWT or FFT.
    """
    def __init__(self):
        super(Layer2_MarginalSegregatedCNN, self).__init__()

        # Slow-wave branch: captures δ and θ activity
        self.slow_wave = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
            # Out: 256 /2 /4 = 32 time steps → (B, 64, 32)
        )

        # Fast-wave branch: captures β and γ activity
        self.fast_wave = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8)
            # Out: 256 /8 = 32 time steps → (B, 64, 32)
        )

        # Fusion FC: (64*32 + 64*32 = 4096) → 128
        self.fc = nn.Sequential(
            nn.Linear(64*32 + 64*32, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        slow = self.slow_wave(x)                            # (B, 64, 32)
        fast = self.fast_wave(x)                            # (B, 64, 32)
        slow = slow.reshape(slow.size(0), -1)               # (B, 2048)
        fast = fast.reshape(fast.size(0), -1)               # (B, 2048)
        combined = torch.cat([slow, fast], dim=1)           # (B, 4096)
        return self.fc(combined)                            # (B, 128)


# ─────────────────────────────────────────────────────────────────
#  LAYER 3: Meta-Classifier (adapted for 128-dim input)
#  Input : (B, 128)  [from Layer 2 only]
#  Output: (B, 2)
# ─────────────────────────────────────────────────────────────────
class Layer3_MetaClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super(Layer3_MetaClassifier, self).__init__()
        self.meta = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.meta(x)


# ─────────────────────────────────────────────────────────────────
#  COMBINED: Layer 2 → Layer 3
# ─────────────────────────────────────────────────────────────────
class Layer2_to_Layer3(nn.Module):
    """
    Ablation Model: Layer 2 (Marginal Segregated CNN) + Layer 3 (Meta-Classifier).
    Layer 1 (CNN-BiLSTM / spatio-temporal pathway) is deliberately omitted
    to measure its isolated contribution to the full architecture.
    """
    def __init__(self, num_classes=2):
        super(Layer2_to_Layer3, self).__init__()
        self.layer2 = Layer2_MarginalSegregatedCNN()
        self.layer3 = Layer3_MetaClassifier(input_dim=128, num_classes=num_classes)

    def forward(self, x):
        feat = self.layer2(x)           # (B, 128)
        return self.layer3(feat)        # (B, 2)


# ─────────────────────────────────────────────────────────────────
#  TRAINING PIPELINE — 10-Fold Stratified CV
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 65)
    print("  ABLATION: Layer 2 → Layer 3  (No Layer 1)")
    print("  Marginal Segregated CNN + Meta-Classifier, No BiLSTM")
    print("=" * 65)

    try:
        X = np.load('X_filtered.npy').astype(np.float32)
        y = np.load('y_train.npy').astype(np.int64)
        print(f"[OK] Data loaded: X={X.shape}, y={y.shape}")
    except FileNotFoundError:
        print("[WARN] Data files not found. Using dummy data.")
        X = np.random.randn(600, 64, 256).astype(np.float32)
        y = np.random.randint(0, 2, 600).astype(np.int64)

    EPOCHS     = 25
    BATCH_SIZE = 64
    K_FOLDS    = 10
    LR         = 1.25e-4
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {DEVICE}")

    avg_train_acc = np.zeros(EPOCHS)
    avg_val_acc   = np.zeros(EPOCHS)
    fold_metrics  = []
    best_global   = 0.0

    skf      = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset  = TensorDataset(X_tensor, y_tensor)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{K_FOLDS}")

        train_loader = DataLoader(Subset(dataset, train_idx), BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   BATCH_SIZE, shuffle=False)

        model     = Layer2_to_Layer3().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        for epoch in range(EPOCHS):
            # ── Train ──
            model.train()
            correct_t, total_t = 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                correct_t += (logits.argmax(1) == yb).sum().item()
                total_t   += yb.size(0)
            t_acc = 100 * correct_t / total_t
            avg_train_acc[epoch] += t_acc

            # ── Validate ──
            model.eval()
            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    probs  = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                    preds  = logits.argmax(1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(yb.cpu().numpy())
                    all_probs.extend(probs)

            v_acc = 100 * accuracy_score(all_labels, all_preds)
            avg_val_acc[epoch] += v_acc

            if v_acc >= best_global:
                best_global = v_acc
                torch.save(model.state_dict(), 'best_model_L23.pth')

        auc  = roc_auc_score(all_labels, all_probs)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)
        fold_metrics.append([v_acc, 100*prec, 100*rec, 100*f1, 100*auc])
        print(f"  Acc={v_acc:.2f}%  Prec={100*prec:.2f}%  Rec={100*rec:.2f}%  "
              f"F1={100*f1:.2f}%  AUC={100*auc:.2f}%")

    avg_train_acc /= K_FOLDS
    avg_val_acc   /= K_FOLDS
    fm = np.array(fold_metrics)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY — Layer 2 → Layer 3  (10-Fold Stratified CV)")
    print("=" * 65)
    for i, lbl in enumerate(["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]):
        print(f"  {lbl:<12}: {fm[:,i].mean():.2f}% ± {fm[:,i].std():.2f}%  "
              f"[Min: {fm[:,i].min():.2f}%  Max: {fm[:,i].max():.2f}%]")
    print(f"  Best Model saved as: best_model_L23.pth")
    print("=" * 65)

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation: Layer 2 → Layer 3  (Freq. Segregation CNN + Meta-Classifier, No BiLSTM)",
                 fontsize=11, fontweight='bold')

    axes[0].plot(range(1, EPOCHS+1), avg_train_acc, 'b-o', ms=4, label='Train Accuracy')
    axes[0].plot(range(1, EPOCHS+1), avg_val_acc,   'r-s', ms=4, label='Val Accuracy')
    axes[0].fill_between(range(1, EPOCHS+1),
                         avg_val_acc - fm[:,0].std(), avg_val_acc + fm[:,0].std(),
                         alpha=0.15, color='red')
    axes[0].set_title('Train vs Validation Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    folds = [f"F{i+1}" for i in range(K_FOLDS)]
    axes[1].bar(folds, fm[:,0], color='darkorange', edgecolor='black', alpha=0.85)
    axes[1].axhline(fm[:,0].mean(), color='red', linestyle='--',
                    label=f'Mean={fm[:,0].mean():.2f}%')
    axes[1].set_title('Per-Fold Validation Accuracy'); axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy (%)'); axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y'); axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('Train_Val_L23.png', dpi=150)
    plt.show()
    print("Plot saved as Train_Val_L23.png")
