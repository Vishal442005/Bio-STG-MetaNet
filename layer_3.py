"""
=============================================================================
  ABLATION CASE 1: LAYER 3 ONLY — Meta-Classifier on Raw EEG
  ============================================================================
  Architecture : Flatten(64x256=16384) → Linear(16384→256) → ReLU → Drop(0.2)
                 → Linear(256→128) → ReLU → Drop(0.2) → Linear(128→64)
                 → ReLU → Drop(0.2) → Linear(64→2)
  Purpose      : Establish a lower-bound baseline — how well does a pure MLP
                 perform on raw EEG without any convolutional or recurrent
                 feature extraction?
  Saves        : best_model_L3.pth, Train_Val_L3.png
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
import os

# ─────────────────────────────────────────────
#  MODEL: Layer 3 — Meta-Classifier MLP Only
#  Input: (B, 64, 256) → flatten → (B, 16384)
# ─────────────────────────────────────────────
class Layer3_Only(nn.Module):
    """
    Pure MLP Meta-Classifier. Operates on raw flattened EEG.
    No spatial (CNN) or temporal (LSTM) preprocessing is applied.
    This represents what Layer 3 alone — without any upstream feature
    extraction — can achieve. Expected accuracy: moderate (~65-75%).
    """
    def __init__(self, input_dim=64*256, num_classes=2):
        super(Layer3_Only, self).__init__()
        self.flatten = nn.Flatten()           # (B, 64, 256) → (B, 16384)
        self.meta_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)        # Raw logits — Softmax applied by CrossEntropy
        )

    def forward(self, x):
        x = self.flatten(x)                   # (B, 16384)
        return self.meta_classifier(x)        # (B, 2)


# ─────────────────────────────────────────────
#  TRAINING PIPELINE — 10-Fold Stratified CV
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 65)
    print("  ABLATION: Layer 3 Only (Meta-Classifier on Raw EEG)")
    print("=" * 65)

    # ── Load Data ──────────────────────────────────────────────────
    try:
        X = np.load('X_filtered.npy').astype(np.float32)
        y = np.load('y_train.npy').astype(np.int64)
        print(f"[OK] Data loaded: X={X.shape}, y={y.shape}")
    except FileNotFoundError:
        print("[WARN] Data files not found. Using dummy data for demonstration.")
        X = np.random.randn(600, 64, 256).astype(np.float32)
        y = np.random.randint(0, 2, 600).astype(np.int64)

    EPOCHS     = 25
    BATCH_SIZE = 64
    K_FOLDS    = 10
    LR         = 1.25e-4
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {DEVICE}")

    # Storage for cross-fold metrics
    avg_train_acc = np.zeros(EPOCHS)
    avg_val_acc   = np.zeros(EPOCHS)
    fold_metrics  = []          # [acc, prec, rec, f1, auc] per fold
    best_global   = 0.0

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset  = TensorDataset(X_tensor, y_tensor)

    print("\n" + "─" * 65)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{K_FOLDS}")

        train_loader = DataLoader(Subset(dataset, train_idx), BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   BATCH_SIZE, shuffle=False)

        model     = Layer3_Only().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        fold_train_accs = []
        fold_val_accs   = []

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
                preds = logits.argmax(1)
                correct_t += (preds == yb).sum().item()
                total_t   += yb.size(0)
            t_acc = 100 * correct_t / total_t
            fold_train_accs.append(t_acc)
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
            fold_val_accs.append(v_acc)
            avg_val_acc[epoch] += v_acc

            if v_acc >= best_global:
                best_global = v_acc
                torch.save(model.state_dict(), 'best_model_L3.pth')

        # Per-fold final metrics
        auc  = roc_auc_score(all_labels, all_probs)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)
        fold_metrics.append([v_acc, 100*prec, 100*rec, 100*f1, 100*auc])
        print(f"  Acc={v_acc:.2f}%  Prec={100*prec:.2f}%  Rec={100*rec:.2f}%  "
              f"F1={100*f1:.2f}%  AUC={100*auc:.2f}%")

    avg_train_acc /= K_FOLDS
    avg_val_acc   /= K_FOLDS

    # ── Summary ────────────────────────────────────────────────────
    fm = np.array(fold_metrics)
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY — Layer 3 Only (10-Fold Stratified CV)")
    print("=" * 65)
    labels_m = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    for i, lbl in enumerate(labels_m):
        print(f"  {lbl:<12}: {fm[:,i].mean():.2f}% ± {fm[:,i].std():.2f}%  "
              f"[Min: {fm[:,i].min():.2f}%  Max: {fm[:,i].max():.2f}%]")
    print(f"  Best Model saved as: best_model_L3.pth")
    print("=" * 65)

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation: Layer 3 Only — Meta-Classifier on Raw EEG", fontsize=13, fontweight='bold')

    axes[0].plot(range(1, EPOCHS+1), avg_train_acc, 'b-o', ms=4, label='Train Accuracy')
    axes[0].plot(range(1, EPOCHS+1), avg_val_acc,   'r-s', ms=4, label='Val Accuracy')
    axes[0].fill_between(range(1, EPOCHS+1), avg_val_acc - fm[:,0].std(),
                         avg_val_acc + fm[:,0].std(), alpha=0.15, color='red')
    axes[0].set_title('Train vs Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    folds = [f"F{i+1}" for i in range(K_FOLDS)]
    axes[1].bar(folds, fm[:,0], color='steelblue', edgecolor='black', alpha=0.8)
    axes[1].axhline(fm[:,0].mean(), color='red', linestyle='--', label=f'Mean={fm[:,0].mean():.2f}%')
    axes[1].set_title('Per-Fold Validation Accuracy')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('Train_Val_L3.png', dpi=150)
    plt.show()
    print("Plot saved as Train_Val_L3.png")
