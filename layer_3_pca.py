"""
=============================================================================
  PCA + LAYER 3 — Principal Component Analysis → Meta-Classifier MLP
  ============================================================================
  Pipeline : Raw EEG (64×256) → Flatten (16384-dim) → PCA (k components,
             retaining ≥95% variance) → Layer 3 Meta-Classifier MLP → 2 classes

  Purpose  : Replicates the spirit of Farsi et al. (2021) Algorithm-1
             (PCA → ANN) but uses our deeper Meta-Classifier (Layer 3) as
             the ANN, giving a fair comparison against Case 1 (L3 on raw EEG).
             The key question answered: "Does linear dimensionality reduction
             (PCA) before Layer 3 help compared to feeding raw EEG directly?"

  Contrasts:
    - Case 1 (layer_3.py)     : Raw 16384-dim → L3 MLP
    - This file (layer_3_pca) : Raw 16384-dim → PCA (k-dim) → L3 MLP

  Saves    : best_model_PCA_L3.pth
             Train_Val_PCA_L3.png
             PCA_Variance_Plot.png  (scree + cumulative explained variance)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PCA Preprocessing (runs once before training, outside the fold)
# ══════════════════════════════════════════════════════════════════════════════
def apply_pca(X_flat, variance_threshold=0.95, verbose=True):
    """
    Fit PCA on the full dataset (before fold splitting) and transform.

    Why fit on full data?
    In a purely unsupervised compression step (PCA finds principal directions
    of variance, not class boundaries), fitting on the complete dataset is
    acceptable and standard practice for EEG dimensionality reduction.
    The transformation is linear and cannot 'leak' class labels.

    Parameters
    ----------
    X_flat            : ndarray (N, 16384) — flattened EEG trials
    variance_threshold: float   — retain enough PCs to explain this fraction
                                  of total variance (default 0.95 = 95%)

    Returns
    -------
    X_pca   : ndarray (N, k)   — PCA-reduced data
    pca     : fitted PCA object (for inspection / later use)
    scaler  : fitted StandardScaler (applied before PCA, required for stability)
    """
    # --- Step 1a: Zero-mean, unit-variance scaling (mandatory before PCA) ---
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # --- Step 1b: Fit full PCA to find all principal components ----------
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # --- Step 1c: Find minimum k that explains ≥ variance_threshold -------
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k      = int(np.searchsorted(cumvar, variance_threshold)) + 1

    if verbose:
        print(f"\n[PCA] Explained variance threshold : {variance_threshold*100:.0f}%")
        print(f"[PCA] Components selected           : {k}  (out of {X_flat.shape[1]})")
        print(f"[PCA] Actual variance retained      : {cumvar[k-1]*100:.2f}%")
        print(f"[PCA] Dimension reduction           : {X_flat.shape[1]} → {k}  "
              f"({100*(1 - k/X_flat.shape[1]):.1f}% reduction)\n")

    # --- Step 1d: Refit PCA retaining only k components -------------------
    pca     = PCA(n_components=k, random_state=42)
    X_pca   = pca.fit_transform(X_scaled)

    return X_pca, pca, scaler, cumvar


def plot_pca_variance(cumvar, k, save_path='PCA_Variance_Plot.png'):
    """
    Plot the scree plot (individual explained variance per component) and the
    cumulative explained variance curve. Marks the selected cutoff at k.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('PCA — Explained Variance Analysis', fontsize=13, fontweight='bold')

    max_show = min(200, len(cumvar))   # Show first 200 components for readability

    # Left: individual explained variance (scree plot)
    individual = np.diff(np.concatenate([[0], cumvar]))
    axes[0].bar(range(1, max_show+1), individual[:max_show] * 100,
                color='steelblue', alpha=0.7, edgecolor='none')
    axes[0].axvline(k, color='red', linestyle='--', label=f'k={k}')
    axes[0].set_title('Scree Plot (Per-Component Variance)')
    axes[0].set_xlabel('Principal Component Index')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Right: cumulative explained variance
    axes[1].plot(range(1, max_show+1), cumvar[:max_show] * 100,
                 color='darkorange', lw=2)
    axes[1].axhline(95, color='gray', linestyle=':', label='95% threshold')
    axes[1].axvline(k, color='red',  linestyle='--', label=f'k={k} selected')
    axes[1].fill_between(range(1, k+1), cumvar[:k] * 100,
                         alpha=0.15, color='darkorange')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[OK] PCA variance plot saved as {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Model: PCA-reduced features → Meta-Classifier MLP (Layer 3)
# ══════════════════════════════════════════════════════════════════════════════
class PCA_Layer3(nn.Module):
    """
    Layer 3 Meta-Classifier receiving PCA-compressed EEG features.

    The MLP architecture is identical to layer_3.py EXCEPT the input
    dimension is k (number of PCA components) instead of 16384.
    This isolates whether PCA's linear compression helps the MLP learn
    better class boundaries than working on raw high-dimensional inputs.

    Mathematical view:
      Raw input  x ∈ R^{16384}
      PCA step   z = V_k^T (x − μ)  ∈ R^k       (linear projection)
      MLP        ŷ = MLP(z)                       (nonlinear classification)
    where V_k ∈ R^{16384×k} are the top-k eigenvectors of the covariance matrix.
    """
    def __init__(self, input_dim, num_classes=2):
        super(PCA_Layer3, self).__init__()
        # Progressive width reduction: input_dim → 256 → 128 → 64 → 2
        # The first hidden size is capped at min(input_dim, 256) so the
        # network still works cleanly when k < 256.
        h1 = min(input_dim, 256)
        h2 = min(h1, 128)
        h3 = min(h2, 64)

        self.meta_classifier = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.44),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.64),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(0.44),

            nn.Linear(h3, num_classes)      # Raw logits → loss handles Softmax
        )

    def forward(self, x):
        return self.meta_classifier(x)      # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — 10-Fold Stratified Training and Evaluation
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 65)
    print("  PCA + Layer 3 — Meta-Classifier on PCA-Compressed EEG")
    print("=" * 65)

    # ── Load Data ──────────────────────────────────────────────────────────
    try:
        X = np.load('X_filtered.npy').astype(np.float32)   # (N, 64, 256)
        y = np.load('y_train.npy').astype(np.int64)
        print(f"[OK] Data loaded: X={X.shape}, y={y.shape}")
    except FileNotFoundError:
        print("[WARN] Data not found. Using synthetic dummy data.")
        X = np.random.randn(600, 64, 256).astype(np.float32)
        y = np.random.randint(0, 2, 600).astype(np.int64)

    # ── Flatten EEG: (N, 64, 256) → (N, 16384) ────────────────────────────
    N          = X.shape[0]
    X_flat     = X.reshape(N, -1)                           # (N, 16384)
    print(f"[INFO] Flattened shape: {X_flat.shape}")

    # ── Apply PCA ──────────────────────────────────────────────────────────
    X_pca, pca_obj, scaler_obj, cumvar = apply_pca(
        X_flat, variance_threshold=0.95, verbose=True
    )
    k = X_pca.shape[1]                                       # Number of PCs selected

    # ── Plot PCA Variance ──────────────────────────────────────────────────
    plot_pca_variance(cumvar, k, save_path='PCA_Variance_Plot.png')

    # ── Hyperparameters ────────────────────────────────────────────────────
    EPOCHS     = 50
    BATCH_SIZE = 32
    K_FOLDS    = 10
    LR         = 1.25e-1
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device     : {DEVICE}")
    print(f"[INFO] PCA dims   : {k}")
    print(f"[INFO] Epochs     : {EPOCHS}")
    print(f"[INFO] Batch size : {BATCH_SIZE}")
    print(f"[INFO] Folds      : {K_FOLDS}")
    print("─" * 65)

    # ── Build Dataset from PCA-reduced features ────────────────────────────
    X_tensor = torch.FloatTensor(X_pca.astype(np.float32))  # (N, k)
    y_tensor = torch.LongTensor(y)
    dataset  = TensorDataset(X_tensor, y_tensor)

    # ── Cross-validation setup ─────────────────────────────────────────────
    avg_train_acc = np.zeros(EPOCHS)
    avg_val_acc   = np.zeros(EPOCHS)
    fold_metrics  = []                  # [acc, prec, rec, f1, auc] per fold
    best_global   = 0.0

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pca, y)):
        print(f"\n  Fold {fold+1}/{K_FOLDS}")

        train_loader = DataLoader(Subset(dataset, train_idx),
                                  BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),
                                  BATCH_SIZE, shuffle=False)

        # Fresh model and optimizer per fold
        model     = PCA_Layer3(input_dim=k).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        for epoch in range(EPOCHS):

            # ── Training Phase ────────────────────────────────────────────
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

            # ── Validation Phase ──────────────────────────────────────────
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

            # Save best model across all folds and epochs
            if v_acc >= best_global:
                best_global = v_acc
                torch.save(model.state_dict(), 'best_model_PCA_L3.pth')

        # ── Per-fold final metrics ────────────────────────────────────────
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = float('nan')
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels,  all_preds, zero_division=0)
        f1   = f1_score(all_labels,      all_preds, zero_division=0)
        fold_metrics.append([v_acc, 100*prec, 100*rec, 100*f1, 100*auc])
        print(f"  Acc={v_acc:.2f}%  Prec={100*prec:.2f}%  Rec={100*rec:.2f}%  "
              f"F1={100*f1:.2f}%  AUC={100*auc:.2f}%")

    # ── Average over folds ─────────────────────────────────────────────────
    avg_train_acc /= K_FOLDS
    avg_val_acc   /= K_FOLDS
    fm = np.array(fold_metrics)

    # ── Results Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  RESULTS SUMMARY — PCA + Layer 3  (10-Fold Stratified CV)")
    print(f"  PCA components used: {k}  (from {X_flat.shape[1]} raw features)")
    print("=" * 65)
    for i, lbl in enumerate(["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]):
        print(f"  {lbl:<12}: {fm[:,i].mean():.2f}% ± {fm[:,i].std():.2f}%  "
              f"[Min={fm[:,i].min():.2f}%  Max={fm[:,i].max():.2f}%]")
    print(f"\n  Best model saved : best_model_PCA_L3.pth")
    print("=" * 65)

    # ── Comparison with Case 1 (raw L3) ───────────────────────────────────
    print("\n  COMPARISON (approximate — run layer_3.py for exact Case 1 values):")
    print(f"  Case 1 (Raw EEG → L3)  : ~82.17% CV  | ~81.17% Live Test")
    print(f"  PCA + L3 (this script) : {fm[:,0].mean():.2f}% ± {fm[:,0].std():.2f}% CV")
    improvement = fm[:,0].mean() - 82.17
    direction   = "BETTER" if improvement > 0 else "WORSE"
    print(f"  Difference             : {improvement:+.2f}%pp  ({direction} than raw L3)")
    print("=" * 65)

    # ── Training / Validation Accuracy Plot ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PCA (k={k}) + Layer 3 Meta-Classifier — 10-Fold Stratified CV",
                 fontsize=12, fontweight='bold')

    epochs_range = range(1, EPOCHS + 1)

    axes[0].plot(epochs_range, avg_train_acc, 'b-o', ms=4, label='Mean Train Accuracy')
    axes[0].plot(epochs_range, avg_val_acc,   'r-s', ms=4, label='Mean Val Accuracy')
    axes[0].fill_between(epochs_range,
                         avg_val_acc - fm[:,0].std(),
                         avg_val_acc + fm[:,0].std(),
                         alpha=0.15, color='red', label='Val ±1 Std')
    axes[0].set_title('Training vs Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    folds_lbl = [f"F{i+1}" for i in range(K_FOLDS)]
    bar_colors = ['steelblue' if acc >= fm[:,0].mean() else 'salmon' for acc in fm[:,0]]
    axes[1].bar(folds_lbl, fm[:,0], color=bar_colors, edgecolor='black', alpha=0.85)
    axes[1].axhline(fm[:,0].mean(), color='green', linestyle='--',
                    label=f'Mean={fm[:,0].mean():.2f}%')
    axes[1].axhline(82.17, color='gray', linestyle=':', lw=1.5,
                    label='Case 1 Raw L3 (82.17%)')
    axes[1].set_title('Per-Fold Validation Accuracy')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('Train_Val_PCA_L3.png', dpi=150)
    plt.show()
    print("[OK] Plot saved as Train_Val_PCA_L3.png")
