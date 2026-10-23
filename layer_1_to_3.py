"""
=============================================================================
  ABLATION CASE 2: LAYER 1 → LAYER 3 (No Layer 2)
  ============================================================================
  Architecture : Layer 1 (CNN-BiLSTM) → 128-dim → Layer 3 (Meta-Classifier)
  Purpose      : Isolate the contribution of spatio-temporal feature extraction
                 (Layer 1) + meta-level decision making (Layer 3), WITHOUT the
                 frequency-segregated pathway (Layer 2).
                 Answers: "Does adding explicit frequency segregation help?"
  Saves        : best_model_L13.pth, Train_Val_L13.png
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

# ─────────────────────────────────────────────────────
#  LAYER 1: Spatio-Temporal CNN-BiLSTM Feature Extractor
#  Input : (B, 64, 256)
#  Output: (B, 128)  ← last hidden state of BiLSTM
# ─────────────────────────────────────────────────────
class Layer1_CNN_BiLSTM(nn.Module):
    """
    Layer 1 — Baseline Spatio-Temporal Extractor.
    Two cascaded Conv1D blocks extract spatial electrode patterns, then a
    Bidirectional LSTM tracks temporal dynamics over the reduced sequence.
    This is identical to Layer 1 of the full Triple-Layer MetaModel.
    """
    def __init__(self):
        super(Layer1_CNN_BiLSTM, self).__init__()

        # Spatial Block: 64ch → 128ch → 256ch, temporal 256 → 64 steps
        self.conv_block = nn.Sequential(
            nn.Conv1d(64,  128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),              # 256 → 128

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),              # 128 → 64
            nn.Dropout(0.2)
        )

        # Temporal Block: BiLSTM over 64 time steps, hidden_size=64 per direction
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=64, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )

    def forward(self, x):
        x = self.conv_block(x)                  # (B, 256, 64)
        x = x.permute(0, 2, 1)                  # (B, 64, 256)
        lstm_out, _ = self.lstm(x)               # (B, 64, 128)
        return lstm_out[:, -1, :]                # (B, 128) — last timestep


# ─────────────────────────────────────────────────────
#  LAYER 3: Meta-Classifier (adapted for 128-dim input)
#  Input : (B, 128)  [from Layer 1 only]
#  Output: (B, 2)
# ─────────────────────────────────────────────────────
class Layer3_MetaClassifier(nn.Module):
    """
    Layer 3 — Meta-Classifier receiving features from Layer 1 only.
    Input dimension is 128 (single pathway) instead of 256 (dual pathway).
    This isolates whether the meta-classifier architecture itself helps,
    independent of Layer 2's frequency segregation.
    """
    def __init__(self, input_dim=128, num_classes=2):
        super(Layer3_MetaClassifier, self).__init__()
        self.meta_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.meta_classifier(x)


# ─────────────────────────────────────────────────────
#  COMBINED: Layer 1 → Layer 3
# ─────────────────────────────────────────────────────
class Layer1_to_Layer3(nn.Module):
    """
    Ablation Model: Layer 1 (CNN-BiLSTM) + Layer 3 (Meta-Classifier).
    Layer 2 (Marginal Segregated CNN / frequency-segregated pathway) is
    deliberately omitted to measure its isolated contribution.
    """
    def __init__(self, num_classes=2):
        super(Layer1_to_Layer3, self).__init__()
        self.layer1 = Layer1_CNN_BiLSTM()
        self.layer3 = Layer3_MetaClassifier(input_dim=128, num_classes=num_classes)

    def forward(self, x):
        feat = self.layer1(x)           # (B, 128)
        return self.layer3(feat)        # (B, 2)


# ─────────────────────────────────────────────────────
#  TRAINING PIPELINE — 10-Fold Stratified CV
# ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 65)
    print("  ABLATION: Layer 1 → Layer 3  (No Layer 2)")
    print("  CNN-BiLSTM + Meta-Classifier, No Freq. Segregation")
    print("=" * 65)

    # ── Load Data ──────────────────────────────────────────────────
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

        model     = Layer1_to_Layer3().to(DEVICE)
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
                torch.save(model.state_dict(), 'best_model_L13.pth')

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
    print("  RESULTS SUMMARY — Layer 1 → Layer 3  (10-Fold Stratified CV)")
    print("=" * 65)
    for i, lbl in enumerate(["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]):
        print(f"  {lbl:<12}: {fm[:,i].mean():.2f}% ± {fm[:,i].std():.2f}%  "
              f"[Min: {fm[:,i].min():.2f}%  Max: {fm[:,i].max():.2f}%]")
    print(f"  Best Model saved as: best_model_L13.pth")
    print("=" * 65)

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation: Layer 1 → Layer 3  (CNN-BiLSTM + Meta-Classifier, No Freq. Seg.)",
                 fontsize=12, fontweight='bold')

    axes[0].plot(range(1, EPOCHS+1), avg_train_acc, 'b-o', ms=4, label='Train Accuracy')
    axes[0].plot(range(1, EPOCHS+1), avg_val_acc,   'r-s', ms=4, label='Val Accuracy')
    axes[0].fill_between(range(1, EPOCHS+1),
                         avg_val_acc - fm[:,0].std(), avg_val_acc + fm[:,0].std(),
                         alpha=0.15, color='red')
    axes[0].set_title('Train vs Validation Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    folds = [f"F{i+1}" for i in range(K_FOLDS)]
    axes[1].bar(folds, fm[:,0], color='seagreen', edgecolor='black', alpha=0.85)
    axes[1].axhline(fm[:,0].mean(), color='red', linestyle='--',
                    label=f'Mean={fm[:,0].mean():.2f}%')
    axes[1].set_title('Per-Fold Validation Accuracy'); axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy (%)'); axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y'); axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('Train_Val_L13.png', dpi=150)
    plt.show()
    print("Plot saved as Train_Val_L13.png")
