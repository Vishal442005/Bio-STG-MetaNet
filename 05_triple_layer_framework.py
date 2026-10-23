"""
FILE: 05_triple_layer_framework.py
PROJECT: Alcoholic EEG Classification - Internship Research
OUR EXTENSION: Triple-Layer Meta-Learning Framework
PURPOSE: Our novel architecture that goes BEYOND Farsi et al.
         Targets >= 95% accuracy by combining:
           Layer 1: Baseline feature vector (PCA + statistical features)
           Layer 2: Deep segregated CNN-LSTM with spatial separation
           Layer 3: Stacking meta-classifier fusing Layer 1 & 2 outputs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────
DATA_DIR = r"C:\Users\Visha\Desktop\Internshhip\data"
OUT_DIR  = r"C:\Users\Visha\Desktop\Internshhip\outputs"

BATCH_SIZE    = 32
EPOCHS_CNN    = 40
LR_CNN        = 5e-4
RANDOM_STATE  = 42
# ────────────────────────────────────────────────────────────


# ══ LAYER 1: Baseline Feature Vector ══════════════════════
def extract_layer1_features(X: np.ndarray) -> np.ndarray:
    """
    Layer 1 — Handcrafted statistical features from time and frequency domains.
    Per channel: mean, std, max, min, skewness, kurtosis, band powers.
    Concatenated across all 64 channels.
    """
    from scipy.stats import skew, kurtosis
    from scipy.signal import welch

    features = []
    for trial in X:
        trial_feats = []
        for ch in range(trial.shape[0]):
            sig = trial[ch, :]
            # Time domain
            trial_feats.extend([
                np.mean(sig), np.std(sig),
                np.max(sig),  np.min(sig),
                float(skew(sig)), float(kurtosis(sig))
            ])
            # Frequency domain (band power)
            f, psd = welch(sig, fs=256, nperseg=64)
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]  # delta, theta, alpha, beta, gamma
            for lo, hi in bands:
                mask = (f >= lo) & (f <= hi)
                trial_feats.append(np.trapz(psd[mask], f[mask]))
        features.append(trial_feats)

    return np.array(features, dtype=np.float32)


# ══ LAYER 2: Deep CNN-LSTM with Spatial Segregation ═══════
class SpatialCNNLSTM(nn.Module):
    """
    Layer 2 deep model:
    - Marginal 1D convolutions for local spatial pattern extraction
    - Bidirectional LSTM for temporal dynamics
    - Returns raw logits AND penultimate embedding for meta-classifier
    """
    def __init__(self, in_channels=64, n_classes=2):
        super().__init__()
        # Spatial CNN block
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),          # 256 → 128
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),          # 128 → 64
            nn.Dropout(0.4)
        )
        # Temporal LSTM block
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )
        self.ln = nn.LayerNorm(256)
        self.embed_dim = 128  # embedding size before final FC

        self.embed_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x, return_embed=False):
        # x: (B, 64, 256)
        x = self.spatial_conv(x)    # (B, 256, 64)
        x = x.permute(0, 2, 1)      # (B, 64, 256)
        out, _ = self.lstm(x)        # (B, 64, 512)
        last = self.ln(out[:, -1, :])  # (B, 512)
        embed = self.embed_head(last)  # (B, 128)
        logits = self.classifier(embed)
        if return_embed:
            return logits, embed
        return logits


# ══ LAYER 3: Stacking Meta-Classifier ═════════════════════
class StackingMetaClassifier:
    """
    Layer 3 — Meta-learning ensemble:
    Combines Layer 1 statistical features with Layer 2 deep embeddings.
    Uses Logistic Regression + Random Forest ensemble for final decision.
    """
    def __init__(self):
        self.lr  = LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_STATE)
        self.rf  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        self.scaler = StandardScaler()
        self.final  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)

    def fit(self, meta_features: np.ndarray, y: np.ndarray):
        """Meta_features = concat(layer1_feats, layer2_embeds)"""
        X_scaled = self.scaler.fit_transform(meta_features)
        self.lr.fit(X_scaled, y)
        self.rf.fit(X_scaled, y)

        # Stacking: use LR + RF proba as input to final classifier
        lr_proba = self.lr.predict_proba(X_scaled)
        rf_proba = self.rf.predict_proba(X_scaled)
        stack_input = np.hstack([lr_proba, rf_proba])
        self.final.fit(stack_input, y)

    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(meta_features)
        lr_proba = self.lr.predict_proba(X_scaled)
        rf_proba = self.rf.predict_proba(X_scaled)
        stack_input = np.hstack([lr_proba, rf_proba])
        return self.final.predict(stack_input)

    def score(self, meta_features, y) -> float:
        return accuracy_score(y, self.predict(meta_features)) * 100


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("  Triple-Layer Meta-Learning Framework")
    print("  Our Novel Extension — Target: >= 95%")
    print("=" * 60)

    data_path = Path(DATA_DIR)
    out_path  = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    X = np.load(data_path / 'X_filtered_fir.npy')
    y = np.load(data_path / 'y_train.npy')

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ─── LAYER 1 ───
    print("\n[Layer 1] Extracting statistical features...")
    L1_tr  = extract_layer1_features(X_tr)
    L1_val = extract_layer1_features(X_val)
    print(f"  Layer 1 feature dim: {L1_tr.shape}")

    # ─── LAYER 2 ───
    print("\n[Layer 2] Training SpatialCNNLSTM...")
    model = SpatialCNNLSTM().to(device)
    train_ld = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_ld = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=BATCH_SIZE
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_CNN, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_CNN)

    best_acc, best_weights = 0.0, None
    for ep in range(1, EPOCHS_CNN + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            correct = sum(
                (model(xb.to(device)).argmax(1) == yb.to(device)).sum().item()
                for xb, yb in val_ld
            )
        acc = correct / len(y_val) * 100
        if acc > best_acc:
            best_acc = acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"  Ep {ep}/{EPOCHS_CNN} | Val Acc: {acc:.2f}%")

    model.load_state_dict(best_weights)
    print(f"  Layer 2 standalone accuracy: {best_acc:.2f}%")

    # Extract embeddings for meta-classifier
    def get_embeds(loader):
        embeds, labels = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                _, emb = model(xb.to(device), return_embed=True)
                embeds.append(emb.cpu().numpy())
                labels.extend(yb.numpy())
        return np.vstack(embeds), np.array(labels)

    L2_tr_emb,  _ = get_embeds(train_ld)
    L2_val_emb, _ = get_embeds(val_ld)

    # ─── LAYER 3 ───
    print("\n[Layer 3] Training stacking meta-classifier...")
    meta_tr  = np.hstack([L1_tr,  L2_tr_emb])
    meta_val = np.hstack([L1_val, L2_val_emb])

    meta_clf = StackingMetaClassifier()
    meta_clf.fit(meta_tr, y_tr)
    final_acc = meta_clf.score(meta_val, y_val)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — Triple-Layer Framework")
    print(f"  Algorithm 1 (PCA+ANN) baseline : 86.00%")
    print(f"  Algorithm 2 (LSTM)    baseline : 93.00%")
    print(f"  Layer 2 CNN-LSTM standalone    : {best_acc:.2f}%")
    print(f"  Layer 3 Meta-Classifier FINAL  : {final_acc:.2f}%")
    print(f"  Target                         : >= 95.00%")
    print(f"  STATUS: {'✓ TARGET MET' if final_acc >= 95 else '↻ IN PROGRESS'}")
    print(f"{'='*60}")

    preds = meta_clf.predict(meta_val)
    print("\nClassification Report (Triple-Layer):")
    print(classification_report(y_val, preds, target_names=['Control', 'Alcoholic']))
