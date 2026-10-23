"""
FILE: 03_algorithm1_pca_ann.py
PROJECT: Alcoholic EEG Classification - Internship Research
PAPER: Farsi et al. (2021) — Algorithm 1
PURPOSE: PCA-based feature extraction followed by ANN classification.
         Replicates Algorithm 1 from the paper (baseline: 86% accuracy).

ALGORITHM 1 PIPELINE:
  Raw EEG → Flatten → PCA (dimensionality reduction)
  → Feature vector → ANN (3-layer MLP) → Softmax → Binary class
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Visha\Desktop\Internshhip\data"
OUT_DIR    = r"C:\Users\Visha\Desktop\Internshhip\outputs"

N_COMPONENTS = 64   # PCA components (as per paper)
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3
HIDDEN_DIMS  = [256, 128, 64]   # ANN architecture
DROPOUT      = 0.4
N_FOLDS      = 5
RANDOM_STATE = 42
# ────────────────────────────────────────────────────────────


class ANNClassifier(nn.Module):
    """
    Artificial Neural Network (ANN) as described in Algorithm 1 of Farsi et al.
    Multi-layer perceptron for binary EEG classification after PCA.

    Architecture:
        Input (PCA features) → Dense(256, ReLU) → BN → Dropout
                            → Dense(128, ReLU) → BN → Dropout
                            → Dense(64,  ReLU) → BN → Dropout
                            → Dense(2) → Softmax
    """
    def __init__(self, input_dim: int, hidden_dims: list = HIDDEN_DIMS,
                 n_classes: int = 2, dropout: float = DROPOUT):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_pca_features(
    X: np.ndarray,
    n_components: int = N_COMPONENTS,
    scaler=None,
    pca=None,
    fit: bool = True
) -> tuple:
    """
    Flatten EEG tensor (N, 64, 256) → (N, 16384),
    standardize, then apply PCA to extract n_components.
    Returns transformed features + fitted scaler + fitted PCA objects.
    """
    N = X.shape[0]
    X_flat = X.reshape(N, -1)  # (N, 64*256=16384)

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_scaled = scaler.transform(X_flat)
        X_pca = pca.transform(X_scaled)

    print(f"  PCA: {X_flat.shape} → {X_pca.shape}")
    if fit:
        evr = pca.explained_variance_ratio_.cumsum()[-1] * 100
        print(f"  Cumulative explained variance: {evr:.2f}%")

    return X_pca, scaler, pca


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total * 100


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    return total_loss / total, correct / total * 100, all_preds, all_labels


def plot_training_curves(history: dict, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1526')
    for ax in (ax1, ax2):
        ax.set_facecolor('#060b18')
        ax.tick_params(colors='#5d7499')
        ax.spines[:].set_color('#1e2d45')

    epochs = range(1, len(history['train_acc']) + 1)

    ax1.plot(epochs, history['train_loss'], '#ef4444', label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   '#f59e0b', label='Val Loss')
    ax1.set_title('Loss Curves — Algorithm 1 (PCA + ANN)', color='white')
    ax1.set_xlabel('Epoch', color='#94a8c4')
    ax1.set_ylabel('Cross-Entropy Loss', color='#94a8c4')
    ax1.legend(facecolor='#0d1526', labelcolor='white')

    ax2.plot(epochs, history['train_acc'], '#00f0ff', label='Train Acc')
    ax2.plot(epochs, history['val_acc'],   '#10d9a0', label='Val Acc')
    ax2.axhline(86, color='#8b5cf6', linestyle='--', alpha=0.7, label='Paper Baseline (86%)')
    ax2.set_title('Accuracy Curves — Algorithm 1', color='white')
    ax2.set_xlabel('Epoch', color='#94a8c4')
    ax2.set_ylabel('Accuracy (%)', color='#94a8c4')
    ax2.legend(facecolor='#0d1526', labelcolor='white')
    ax2.set_ylim([40, 100])

    fig.suptitle('Algorithm 1: PCA Feature Extraction + ANN Classification', color='white', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0d1526')
    sns.heatmap(
        cm, annot=True, fmt='d', ax=ax,
        cmap='Blues',
        xticklabels=['Control', 'Alcoholic'],
        yticklabels=['Control', 'Alcoholic'],
        cbar_kws={'label': 'Count'}
    )
    ax.set_title(title, color='white', pad=12)
    ax.set_xlabel('Predicted', color='#94a8c4')
    ax.set_ylabel('True Label', color='#94a8c4')
    ax.set_facecolor('#060b18')
    fig.set_facecolor('#0d1526')
    ax.tick_params(colors='white')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("  Algorithm 1: PCA + ANN  (Farsi et al. 2021)")
    print("=" * 60)

    data_path = Path(DATA_DIR)
    out_path  = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    X = np.load(data_path / 'X_filtered_fir.npy')
    y = np.load(data_path / 'y_train.npy')
    print(f"Data: {X.shape}, Labels: {y.shape}")

    # PCA Feature Extraction
    print("\nExtracting PCA features...")
    X_pca, scaler, pca = extract_pca_features(X, n_components=N_COMPONENTS)

    # Train/Val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pca, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    val_ds   = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = ANNClassifier(input_dim=N_COMPONENTS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    print("\nTraining...")
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_ld, criterion, optimizer, device)
        vl_loss, vl_acc, preds, labels = eval_epoch(model, val_ld, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), str(out_path / 'alg1_best_model.pth'))

        if ep % 10 == 0:
            print(f"Ep {ep:3d}/{EPOCHS} | TrainLoss: {tr_loss:.4f} | ValAcc: {vl_acc:.2f}%")

    print(f"\n>>> Best Validation Accuracy (Algorithm 1): {best_acc:.2f}%")
    print(f">>> Paper baseline: 86.00%")
    print(f">>> Gap from paper: {best_acc - 86:.2f}%")

    plot_training_curves(history, save_path=str(out_path / 'alg1_training_curves.png'))

    # Final evaluation
    model.load_state_dict(torch.load(str(out_path / 'alg1_best_model.pth')))
    _, _, final_preds, final_labels = eval_epoch(model, val_ld, criterion, device)
    print("\nClassification Report (Algorithm 1):")
    print(classification_report(final_labels, final_preds, target_names=['Control', 'Alcoholic']))

    plot_confusion_matrix(
        final_labels, final_preds,
        title='Algorithm 1: PCA + ANN — Confusion Matrix',
        save_path=str(out_path / 'alg1_confusion_matrix.png')
    )
