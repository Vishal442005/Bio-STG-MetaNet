"""
COMPLETE EEG CLASSIFICATION PIPELINE
=====================================
Author: Vishal Chandel (Internship Research)
Paper: Enhanced Spatio-Temporal Meta-Model for Classification of Alcoholic EEG Signals

This is the COMPLETE version that runs everything in one go:
- Full dataset training with 10-fold cross-validation
- Comprehensive ablation study (8 configurations)
- Baseline model comparison (4 models)
- Cross-dataset validation (bidirectional)
- Statistical significance analysis
- All plots and visualizations
- LaTeX table generation
- Paper update

All outputs: plots/, results/, models/
"""

import os, sys, time, copy, gzip, glob, random, warnings, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Force output to appear immediately
sys.stdout.reconfigure(line_buffering=True)
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy import signal, stats
from scipy.stats import ttest_rel, ttest_ind, wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CFG = {
    # -- Dataset paths ------------------------------------------------------
    'train_dir':    'SMNI_CMI_TRAIN',
    'test_dir':     'SMNI_CMI_TEST',
    'eeg_full_dir': 'eeg_full',

    # -- Dataset sizes -------------------------------------------------------
    'full_files': None,          # None = all available files
    'full_folds': 10,
    'full_epochs': 50,

    # -- Model hyper-params -------------------------------------------------
    'batch_size': 32,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    'grad_clip': 1.0,
    'patience': 12,              # early stopping patience
    'lr_patience': 5,            # ReduceLROnPlateau patience
    'seed': 42,

    # -- Output dirs --------------------------------------------------------
    'plots_dir':   'plots',
    'results_dir': 'results',
    'models_dir':  'models',
}

# -- Reproducibility ---------------------------------------------------------
random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE}", flush=True)
print("[INFO] Setting up directories...", flush=True)

for d in [CFG['plots_dir'], CFG['results_dir'], CFG['models_dir']]:
    os.makedirs(d, exist_ok=True)

print("[INFO] Setup complete. Ready to start training.", flush=True)

# ==============================================================================
# 1.  DATA LOADING & PREPROCESSING
# ==============================================================================

def parse_gz_file(path):
    """Parse one SMNI .gz EEG file -> (64, 256) float32 array or None."""
    try:
        with gzip.open(path, 'rt') as fh:
            lines = fh.readlines()

        data = []
        channels_seen = set()
        channel_to_idx = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                ch_name  = parts[1]
                samp_idx = int(parts[2])
                voltage  = float(parts[3])
                data.append((ch_name, samp_idx, voltage))
                channels_seen.add(ch_name)
            except (ValueError, IndexError):
                continue

        if len(data) == 0:
            return None

        # Build ordered channel index
        sorted_channels = sorted(channels_seen)
        ch_idx = {ch: i for i, ch in enumerate(sorted_channels)}
        n_ch = min(len(sorted_channels), 64)

        mat = np.zeros((64, 256), dtype=np.float32)
        for ch_name, samp_idx, voltage in data:
            i = ch_idx.get(ch_name, -1)
            if 0 <= i < n_ch and 0 <= samp_idx < 256:
                mat[i, samp_idx] = voltage

        return mat
    except Exception:
        return None


def butterworth_bandpass(X, fs=256, lo=0.5, hi=45.0, order=4):
    """Zero-phase Butterworth bandpass filter applied per channel per trial."""
    nyq = fs / 2.0
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype='band')
    Xf = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xf[i, j] = signal.filtfilt(b, a, X[i, j])
    return Xf


def zscore_normalize(X):
    """Per-channel per-trial z-score normalization."""
    Xn = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mu, sd = X[i, j].mean(), X[i, j].std()
            Xn[i, j] = (X[i, j] - mu) / (sd + 1e-8)
    return Xn


def collect_file_paths(train_dir='SMNI_CMI_TRAIN', test_dir='SMNI_CMI_TEST',
                       max_files=None):
    """Collect balanced (alc/ctrl) .gz file paths."""
    alc, ctrl = [], []
    for d in [train_dir, test_dir]:
        if not os.path.isdir(d):
            continue
        for subj in os.listdir(d):
            spath = os.path.join(d, subj)
            if not os.path.isdir(spath):
                continue
            label = None
            if 'co2a' in subj.lower():
                label = 1
            elif 'co2c' in subj.lower():
                label = 0
            if label is None:
                continue
            for gz in glob.glob(os.path.join(spath, '*.gz')):
                if label == 1:
                    alc.append(gz)
                else:
                    ctrl.append(gz)

    random.shuffle(alc)
    random.shuffle(ctrl)

    if max_files is not None:
        n = max_files // 2
        alc  = alc[:n]
        ctrl = ctrl[:n]
    else:
        n = min(len(alc), len(ctrl))
        alc  = alc[:n]
        ctrl = ctrl[:n]

    paths = alc + ctrl
    labels = [1] * len(alc) + [0] * len(ctrl)
    print(f"[DATA] Collected {len(alc)} alcoholic + {len(ctrl)} control "
          f"= {len(paths)} total files")
    return paths, labels


def load_dataset(max_files=None, train_dir='SMNI_CMI_TRAIN',
                 test_dir='SMNI_CMI_TEST', cache_prefix=None):
    """Load, filter, normalize EEG data. Returns (X, y) numpy arrays."""
    # Cache check
    if cache_prefix:
        cx = os.path.join(CFG['results_dir'], f'{cache_prefix}_X.npy')
        cy = os.path.join(CFG['results_dir'], f'{cache_prefix}_y.npy')
        if os.path.exists(cx) and os.path.exists(cy):
            X = np.load(cx)
            y = np.load(cy)
            print(f"[DATA] Loaded cached {cache_prefix}: X={X.shape}, y={y.shape}")
            return X, y

    paths, labels = collect_file_paths(train_dir, test_dir, max_files)

    print(f"[DATA] Parsing {len(paths)} files …")
    X_list, y_list = [], []
    for idx, (p, lbl) in enumerate(zip(paths, labels)):
        mat = parse_gz_file(p)
        if mat is not None:
            X_list.append(mat)
            y_list.append(lbl)
        if (idx + 1) % 50 == 0:
            print(f"  [DATA] {idx+1}/{len(paths)} files parsed …")

    X = np.stack(X_list, axis=0)   # (N, 64, 256)
    y = np.array(y_list, dtype=np.int64)
    print(f"[DATA] Raw: {X.shape}  Alc={y.sum()}  Ctrl={(y==0).sum()}")

    print("[DATA] Applying bandpass filter (0.5–45 Hz) …")
    X = butterworth_bandpass(X)
    print("[DATA] Applying z-score normalisation …")
    X = zscore_normalize(X)

    if cache_prefix:
        np.save(cx, X)
        np.save(cy, y)
        print(f"[DATA] Cached to {cx}, {cy}")

    return X, y


# ==============================================================================
# 2.  MODEL ARCHITECTURE  (Enhanced Triple-Layer MetaModel)
# ==============================================================================

class SEBlock(nn.Module):
    """Squeeze-Excitation channel attention."""
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // r),
            nn.ReLU(),
            nn.Linear(ch // r, ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, k, padding=k//2, bias=False),
            nn.BatchNorm1d(ch_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch_out, ch_out, k, padding=k//2, bias=False),
            nn.BatchNorm1d(ch_out),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, 1, bias=False),
            nn.BatchNorm1d(ch_out)
        ) if ch_in != ch_out else nn.Identity()
        self.se = SEBlock(ch_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.conv(x)) + self.skip(x))


class Layer1_SpatioTemporal(nn.Module):
    """CNN -> SE -> BiLSTM -> Multi-Head Attention."""
    def __init__(self, out_dim=192):
        super().__init__()
        self.cnn = nn.Sequential(
            ResidualBlock(64, 128, k=7, dropout=0.15),
            nn.MaxPool1d(2),
            ResidualBlock(128, 256, k=5, dropout=0.15),
            nn.MaxPool1d(2),
            ResidualBlock(256, 256, k=3, dropout=0.1),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1,
                                          batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim), nn.GELU(), nn.Dropout(0.15)
        )

    def forward(self, x):                  # x: (B,64,256)
        h = self.cnn(x)                    # (B,256,64)
        h = h.permute(0, 2, 1)            # (B,64,256)
        h, _ = self.lstm(h)               # (B,64,256)
        h_attn, _ = self.attn(h, h, h)   # (B,64,256)
        h = self.norm(h + h_attn)
        h = h.mean(dim=1)                 # global average (B,256)
        return self.proj(h)               # (B,out_dim)


class Layer2_FrequencySegregated(nn.Module):
    """Parallel slow/mid/fast streams with SE + residuals."""
    def __init__(self, out_dim=192):
        super().__init__()
        def stream(k):
            return nn.Sequential(
                nn.Conv1d(64, 64, k, padding=k//2, bias=False),
                nn.BatchNorm1d(64), nn.GELU(),
                ResidualBlock(64, 128, k=3, dropout=0.15),
                nn.AdaptiveAvgPool1d(16),
            )
        self.slow = stream(31)   # captures δ/θ
        self.mid  = stream(11)   # captures α/β
        self.fast = stream(3)    # captures β/γ

        self.fuse = nn.Sequential(
            nn.Linear(128 * 16 * 3, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, out_dim), nn.GELU(), nn.Dropout(0.1),
        )

    def forward(self, x):                  # x: (B,64,256)
        s = self.slow(x).reshape(x.size(0), -1)
        m = self.mid(x).reshape(x.size(0), -1)
        f = self.fast(x).reshape(x.size(0), -1)
        return self.fuse(torch.cat([s, m, f], dim=1))


class StatFeatureExtractor(nn.Module):
    """Statistical features: mean, std, max, min, skewness, kurtosis per channel."""
    def __init__(self, out_dim=64):
        super().__init__()
        # 64 channels × 6 stats = 384
        self.proj = nn.Sequential(
            nn.Linear(64 * 6, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim), nn.GELU(),
        )

    @staticmethod
    def _skew(x):
        mu = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + 1e-8
        return ((x - mu) / std).pow(3).mean(-1)

    @staticmethod
    def _kurt(x):
        mu = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + 1e-8
        return ((x - mu) / std).pow(4).mean(-1)

    def forward(self, x):                  # x: (B,64,256)
        feats = torch.cat([
            x.mean(-1), x.std(-1),
            x.max(-1).values, x.min(-1).values,
            self._skew(x), self._kurt(x),
        ], dim=-1)                          # (B, 64*6)
        return self.proj(feats)


class Layer3_MetaClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.cls(x * self.gate(x))


class EnhancedMetaModel(nn.Module):
    """
    Full Enhanced Triple-Layer MetaModel:
      Layer1: Spatio-temporal (CNN + BiLSTM + MHA)
      Layer2: Frequency-segregated (slow/mid/fast streams)
      StatExtractor: 6-stat features per channel
      Layer3: Gated meta-classifier
    """
    def __init__(self, num_classes=2, l1_dim=192, l2_dim=192, stat_dim=64):
        super().__init__()
        self.layer1 = Layer1_SpatioTemporal(out_dim=l1_dim)
        self.layer2 = Layer2_FrequencySegregated(out_dim=l2_dim)
        self.stats  = StatFeatureExtractor(out_dim=stat_dim)
        in_dim = l1_dim + l2_dim + stat_dim
        self.layer3 = Layer3_MetaClassifier(in_dim, num_classes)

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(x)
        fs = self.stats(x)
        return self.layer3(torch.cat([f1, f2, fs], dim=1))

    def get_embeddings(self, x):
        with torch.no_grad():
            return {
                'spatio_temporal': self.layer1(x),
                'frequency': self.layer2(x),
                'statistical': self.stats(x),
            }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# 3.  ABLATION MODEL VARIANTS
# ==============================================================================

class AblationVariant(nn.Module):
    """Ablation model that enables/disables sub-modules."""
    def __init__(self, use_l1=True, use_l2=True, use_stats=True,
                 use_attention=True, use_residual=True, num_classes=2):
        super().__init__()
        self.use_l1 = use_l1
        self.use_l2 = use_l2
        self.use_stats = use_stats

        dims = []
        if use_l1:
            if use_attention and use_residual:
                self.layer1 = Layer1_SpatioTemporal(192)
                dims.append(192)
            elif use_residual:
                # No attention, simple CNN+LSTM
                self.layer1 = _SimpleL1(192)
                dims.append(192)
            else:
                # No residual, no attention
                self.layer1 = _PlainL1(128)
                dims.append(128)
        if use_l2:
            self.layer2 = Layer2_FrequencySegregated(192)
            dims.append(192)
        if use_stats:
            self.stats = StatFeatureExtractor(64)
            dims.append(64)

        in_dim = sum(dims) if dims else 64
        if in_dim == 0:
            in_dim = 128   # fallback
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        parts = []
        if self.use_l1:
            parts.append(self.layer1(x))
        if self.use_l2:
            parts.append(self.layer2(x))
        if self.use_stats:
            parts.append(self.stats(x))
        if not parts:
            # degenerate: global avg
            parts.append(x.mean(-1).mean(-1, keepdim=True).expand(-1, 128))
        return self.head(torch.cat(parts, dim=1))


class _SimpleL1(nn.Module):
    """L1 without multi-head attention."""
    def __init__(self, out_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            ResidualBlock(64, 128, k=7), nn.MaxPool1d(2),
            ResidualBlock(128, 256, k=5), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.proj = nn.Sequential(nn.Linear(256, out_dim), nn.GELU())

    def forward(self, x):
        h = self.cnn(x).permute(0, 2, 1)
        h, _ = self.lstm(h)
        return self.proj(h.mean(1))


class _PlainL1(nn.Module):
    """L1 with plain CNN (no residual, no attention)."""
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(64, 128, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


ABLATION_CONFIGS = {
    'Full Model (Proposed)': dict(use_l1=True,  use_l2=True,  use_stats=True,
                                   use_attention=True, use_residual=True),
    'w/o Layer 1 (No Spatio-Temporal)':
                              dict(use_l1=False, use_l2=True,  use_stats=True,
                                   use_attention=True, use_residual=True),
    'w/o Layer 2 (No Freq-Seg)':
                              dict(use_l1=True,  use_l2=False, use_stats=True,
                                   use_attention=True, use_residual=True),
    'w/o Statistical Features':
                              dict(use_l1=True,  use_l2=True,  use_stats=False,
                                   use_attention=True, use_residual=True),
    'w/o Attention':          dict(use_l1=True,  use_l2=True,  use_stats=True,
                                   use_attention=False, use_residual=True),
    'w/o Residual Connections':
                              dict(use_l1=True,  use_l2=True,  use_stats=True,
                                   use_attention=True, use_residual=False),
    'Layer 1 Only':           dict(use_l1=True,  use_l2=False, use_stats=False,
                                   use_attention=True, use_residual=True),
    'Layer 2 Only':           dict(use_l1=False, use_l2=True,  use_stats=False,
                                   use_attention=True, use_residual=True),
}


# ==============================================================================
# 4.  BASELINE MODELS
# ==============================================================================

class PCA_ANN_Baseline(nn.Module):
    """Farsi et al. 2021 – PCA + ANN (simulated as linear projection + MLP)."""
    def __init__(self, n_pca=50, num_classes=2):
        super().__init__()
        self.pca_proj = nn.Linear(64 * 256, n_pca, bias=False)
        self.ann = nn.Sequential(
            nn.Linear(n_pca, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        flat = x.reshape(x.size(0), -1)
        return self.ann(self.pca_proj(flat))


class VanillaLSTM_Baseline(nn.Module):
    """Vanilla BiLSTM baseline."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.cls = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        h, _ = self.lstm(x.permute(0, 2, 1))  # (B,256,64)->(B,256,128*2)
        return self.cls(h[:, -1, :])


class VanillaCNN_Baseline(nn.Module):
    """Vanilla CNN baseline."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class CNN_LSTM_Baseline(nn.Module):
    """CNN-LSTM hybrid baseline (Li & Wu 2022 style)."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.cls = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        h = self.cnn(x).permute(0, 2, 1)
        h, _ = self.lstm(h)
        return self.cls(h[:, -1, :])


BASELINE_MODELS = {
    'PCA + ANN (Farsi 2021)':       PCA_ANN_Baseline,
    'Vanilla BiLSTM':               VanillaLSTM_Baseline,
    'Vanilla CNN':                  VanillaCNN_Baseline,
    'CNN-LSTM Hybrid (Li 2022)':    CNN_LSTM_Baseline,
}


# ==============================================================================
# 5.  TRAINING ENGINE
# ==============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        n_cls = logits.size(1)
        smooth = self.ls / (n_cls - 1)
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.ls) + smooth * (1 - one_hot)
        log_p = F.log_softmax(logits, dim=1)
        ce = -(one_hot * log_p).sum(dim=1)
        p = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1 - p).pow(self.gamma)
        return (focal_w * ce).mean()


def make_loader(X, y, batch_size=32, shuffle=True, balance=False):
    """Create DataLoader from numpy arrays, optionally with balanced sampling."""
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    ds = TensorDataset(Xt, yt)
    if balance and shuffle:
        cls_cnt = np.bincount(y)
        weights = 1.0 / cls_cnt[y]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_epoch(model, loader, optimizer, criterion, grad_clip=1.0):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_preds, all_probs, all_labels = [], [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        out = model(Xb)
        loss = criterion(out, yb)
        probs = F.softmax(out, dim=1)[:, 1]
        total_loss += loss.item() * len(yb)
        preds = out.argmax(1)
        correct += (preds == yb).sum().item()
        n += len(yb)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(yb.cpu().tolist())
    acc = correct / n
    return total_loss / n, acc, all_preds, all_probs, all_labels


def train_model_cv(model_fn, X, y, n_folds=5, epochs=40,
                   lr=3e-4, weight_decay=1e-4, batch_size=32,
                   patience=12, lr_patience=5, grad_clip=1.0,
                   label_smoothing=0.1, save_path=None, tag='model'):
    """
    Stratified k-fold cross-validation trainer.
    Returns dict with fold metrics, histories, best model state.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CFG['seed'])
    criterion = FocalLoss(gamma=2.0, label_smoothing=label_smoothing)

    fold_metrics = []
    all_histories = []
    best_val_acc_overall = 0.0
    best_state = None
    all_true, all_pred, all_prob = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'-'*60}")
        print(f"  FOLD {fold}/{n_folds} — {tag}")
        print(f"{'-'*60}")

        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        tr_loader = make_loader(X_tr, y_tr, batch_size, shuffle=True, balance=True)
        vl_loader = make_loader(X_vl, y_vl, batch_size, shuffle=False)

        model = model_fn().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=lr_patience)

        best_val_acc = 0.0
        best_model_state = None
        patience_ctr = 0
        history = {'train_loss': [], 'val_loss': [],
                   'train_acc': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_epoch(model, tr_loader, optimizer,
                                        criterion, grad_clip)
            v_loss, v_acc, vpreds, vprobs, vlabels = eval_epoch(
                model, vl_loader, criterion)
            scheduler.step(v_acc)

            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                if v_acc > best_val_acc_overall:
                    best_val_acc_overall = v_acc
                    best_state = copy.deepcopy(model.state_dict())
            else:
                patience_ctr += 1

            # Print progress every epoch to ensure visibility
            print(f"    Epoch {epoch:3d}/{epochs}: "
                  f"TrainLoss={t_loss:.4f} TrainAcc={t_acc*100:.1f}%  "
                  f"ValLoss={v_loss:.4f} ValAcc={v_acc*100:.2f}%  "
                  f"BestValAcc={best_val_acc*100:.2f}%", flush=True)

            if patience_ctr >= patience:
                print(f"    [EarlyStopping] at epoch {epoch}")
                break

        # Load best fold model and evaluate
        model.load_state_dict(best_model_state)
        _, _, vpreds, vprobs, vlabels = eval_epoch(model, vl_loader, criterion)

        prec  = precision_score(vlabels, vpreds, zero_division=0)
        rec   = recall_score(vlabels, vpreds, zero_division=0)
        f1    = f1_score(vlabels, vpreds, zero_division=0)
        try:
            auc_roc = roc_auc_score(vlabels, vprobs)
        except Exception:
            auc_roc = 0.5

        fold_metrics.append({
            'fold': fold,
            'val_acc': best_val_acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc_roc': auc_roc,
        })
        all_histories.append(history)
        all_true.extend(vlabels)
        all_pred.extend(vpreds)
        all_prob.extend(vprobs)

        print(f"  Fold {fold} Best Val Acc: {best_val_acc*100:.2f}%  "
              f"AUC-ROC: {auc_roc:.4f}")

    # Save best model
    if save_path and best_state:
        torch.save(best_state, save_path)
        print(f"[MODEL] Saved best model to {save_path}")

    df = pd.DataFrame(fold_metrics)
    return {
        'fold_df': df,
        'histories': all_histories,
        'all_true': all_true,
        'all_pred': all_pred,
        'all_prob': all_prob,
        'mean_acc': df['val_acc'].mean(),
        'std_acc':  df['val_acc'].std(),
        'mean_auc': df['auc_roc'].mean(),
        'std_auc':  df['auc_roc'].std(),
    }


# ==============================================================================
# 6.  INFERENCE LATENCY MEASUREMENT
# ==============================================================================

@torch.no_grad()
def measure_latency(model, n_runs=100, batch_size=1):
    """Measure per-sample inference latency in ms."""
    model.eval()
    dummy = torch.randn(batch_size, 64, 256).to(DEVICE)
    # Warm-up
    for _ in range(10):
        _ = model(dummy)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


# ==============================================================================
# 7.  STATISTICAL ANALYSIS
# ==============================================================================

def paired_ttest(a, b):
    """Paired t-test, returns (t_stat, p_value, cohen_d)."""
    a, b = np.array(a), np.array(b)
    # Handle arrays of different lengths by taking the minimum length
    min_len = min(len(a), len(b))
    if min_len > 1:
        a, b = a[:min_len], b[:min_len]
        t, p = ttest_rel(a, b)
        d = (a.mean() - b.mean()) / (np.std(a - b) + 1e-10)
    else:
        # If not enough samples, use unpaired test
        t, p = ttest_ind(a, b)
        d = (a.mean() - b.mean()) / (np.sqrt((np.var(a) + np.var(b))/2) + 1e-10)
    return t, p, d


def compute_95ci(values):
    """95% confidence interval via t-distribution."""
    n = len(values)
    m = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.ppf(0.975, df=n-1) * se
    return m, ci


def full_statistical_report(proposed_accs, baseline_dict):
    """
    Run paired t-tests and effect-size analysis between proposed and all baselines.
    Returns a DataFrame.
    """
    rows = []
    for name, bl_accs in baseline_dict.items():
        t, p, d = paired_ttest(proposed_accs, bl_accs)
        m_prop, ci_prop = compute_95ci(proposed_accs)
        m_bl,   ci_bl   = compute_95ci(bl_accs)
        rows.append({
            'Comparison': f'Proposed vs {name}',
            'Proposed Acc (%)': f'{m_prop*100:.2f} ± {ci_prop*100:.2f}',
            'Baseline Acc (%)': f'{m_bl*100:.2f} ± {ci_bl*100:.2f}',
            't-statistic': round(t, 4),
            'p-value': round(p, 5),
            'Significant (p<0.05)': 'Yes' if p < 0.05 else 'No',
            "Cohen's d": round(d, 4),
            'Effect Size': ('Large' if abs(d) > 0.8 else
                            'Medium' if abs(d) > 0.5 else 'Small'),
        })
    return pd.DataFrame(rows)


# ==============================================================================
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
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
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
                f"d = {d:.2f}\n({effect_size})\n{sig_text}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=12)
    ax.set_ylabel('$-\log_{10}(p	ext{-value})$')
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

# 9.  QUICK TRAIN HELPER (for ablation / baseline with fewer folds/epochs)
# ==============================================================================

def quick_train_cv(model_fn, X, y, n_folds=3, epochs=20,
                   lr=3e-4, weight_decay=1e-4, batch_size=32,
                   patience=8, label_smoothing=0.05):
    """Lighter CV for ablation / baseline runs."""
    print(f"    [QUICK TRAIN] {n_folds}-fold CV, {epochs} epochs...", flush=True)
    result = train_model_cv(
        model_fn, X, y,
        n_folds=n_folds, epochs=epochs, lr=lr,
        weight_decay=weight_decay, batch_size=batch_size,
        patience=patience, lr_patience=4, grad_clip=1.0,
        label_smoothing=label_smoothing,
        save_path=None, tag='quick'
    )
    print(f"    [QUICK TRAIN] Completed. Mean Val Acc: {result['mean_acc']*100:.2f}%", flush=True)
    return result


# ==============================================================================
# 10.  CROSS-DATASET VALIDATION
# ==============================================================================

@torch.no_grad()
def evaluate_on_loader(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n, all_p, all_pr, all_l = 0, 0, [], [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        out = model(Xb)
        probs = F.softmax(out, 1)[:, 1]
        all_p.extend(out.argmax(1).cpu().tolist())
        all_pr.extend(probs.cpu().tolist())
        all_l.extend(yb.cpu().tolist())
        n += len(yb)
    acc = accuracy_score(all_l, all_p)
    try:
        auc_v = roc_auc_score(all_l, all_pr)
    except Exception:
        auc_v = 0.5
    return acc, auc_v, all_l, all_p, all_pr


def cross_dataset_validate(train_X, train_y, test_X, test_y,
                            epochs=30, lr=3e-4, tag='cross_dataset'):
    """Train on train set, evaluate on test set."""
    model = EnhancedMetaModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(label_smoothing=0.1)

    tr_loader = make_loader(train_X, train_y, CFG['batch_size'],
                            shuffle=True, balance=True)
    te_loader = make_loader(test_X,  test_y,  CFG['batch_size'], shuffle=False)

    best_acc, best_state = 0, None
    for ep in range(1, epochs + 1):
        train_epoch(model, tr_loader, optimizer, criterion)
        scheduler.step()
        acc, auc_v, *_ = evaluate_on_loader(model, te_loader)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        if ep % 5 == 0:
            print(f"  [{tag}] Ep {ep}/{epochs}  TestAcc={acc*100:.2f}%")

    model.load_state_dict(best_state)
    return evaluate_on_loader(model, te_loader)


def run_cross_dataset(train_X, train_y, test_X, test_y):
    """Run bidirectional cross-dataset validation."""
    print("\n" + "="*70)
    print("  CROSS-DATASET VALIDATION")
    print("="*70)

    results = {}

    print("\n-> Train on TRAIN set -> Evaluate on TEST set")
    acc, auc_v, y_true, y_pred, y_prob = cross_dataset_validate(
        train_X, train_y, test_X, test_y,
        epochs=30, tag='Train->Test'
    )
    results['Train->Test'] = {'acc': acc, 'auc': auc_v}
    print(f"  Accuracy: {acc*100:.2f}%  AUC-ROC: {auc_v:.4f}")

    print("\n-> Train on TEST set -> Evaluate on TRAIN set")
    acc2, auc2, y_true2, y_pred2, y_prob2 = cross_dataset_validate(
        test_X, test_y, train_X, train_y,
        epochs=30, tag='Test->Train'
    )
    results['Test->Train'] = {'acc': acc2, 'auc': auc2}
    print(f"  Accuracy: {acc2*100:.2f}%  AUC-ROC: {auc2:.4f}")

    domain_gap = abs(acc - acc2) * 100
    print(f"\n  Domain Adaptation Gap: {domain_gap:.2f}%")

    cross_df = pd.DataFrame([
        {'Scenario': 'Train->Test', 'Accuracy (%)': round(acc*100, 2),
         'AUC-ROC': round(auc_v, 4)},
        {'Scenario': 'Test->Train', 'Accuracy (%)': round(acc2*100, 2),
         'AUC-ROC': round(auc2, 4)},
        {'Scenario': 'Domain Gap', 'Accuracy (%)': round(domain_gap, 2),
         'AUC-ROC': '—'},
    ])
    csv_path = os.path.join(CFG['results_dir'], 'cross_dataset_results.csv')
    cross_df.to_csv(csv_path, index=False)
    print(f"[RESULTS] Saved {csv_path}")

    plot_cross_dataset(results, tag='cross_dataset')
    return results, cross_df


# ==============================================================================
# 11.  MASTER RUN FUNCTIONS
# ==============================================================================

def run_full_training(X, y):
    """Run full dataset training with 10-fold cross-validation."""
    print("\n" + "="*70)
    print("  FULL DATASET TRAINING (10-Fold Cross-Validation)")
    print("="*70)

    result = train_model_cv(
        model_fn=EnhancedMetaModel,
        X=X, y=y,
        n_folds=CFG['full_folds'],
        epochs=CFG['full_epochs'],
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay'],
        batch_size=CFG['batch_size'],
        patience=CFG['patience'],
        lr_patience=CFG['lr_patience'],
        grad_clip=CFG['grad_clip'],
        label_smoothing=CFG['label_smoothing'],
        save_path=os.path.join(CFG['models_dir'], 'full_best_model.pth'),
        tag='full',
    )

    # Summary
    df = result['fold_df']
    print("\n[FULL TRAINING RESULTS]")
    print(df.to_string(index=False))
    mean_acc = df['val_acc'].mean() * 100
    std_acc  = df['val_acc'].std()  * 100
    mean_auc = df['auc_roc'].mean() * 100
    std_auc  = df['auc_roc'].std()  * 100
    print(f"\nOverall  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Overall  AUC-ROC:  {mean_auc:.2f}% ± {std_auc:.2f}%")

    # Latency
    model = EnhancedMetaModel().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(CFG['models_dir'], 'full_best_model.pth'),
        map_location=DEVICE))
    lat_mean, lat_std = measure_latency(model)
    print(f"Inference Latency: {lat_mean:.2f} ± {lat_std:.2f} ms/sample")

    # Plots
    plot_training_curves(result['histories'], tag='full',
                         n_folds=CFG['full_folds'])
    plot_confusion_matrix(result['all_true'], result['all_pred'], tag='full')
    plot_roc_curve(result['all_true'], result['all_prob'], tag='full')
    plot_fold_accuracy(df, tag='full')

    # Save CSV
    csv_path = os.path.join(CFG['results_dir'], 'full_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[RESULTS] Saved {csv_path}")

    return result, lat_mean, lat_std


def run_ablation_study(X, y, tag='full'):
    """Run ablation study over all configurations."""
    print("\n" + "="*70)
    print("  ABLATION STUDY (8 Configurations)")
    print("="*70)

    ablation_rows = []
    all_accs = {}
    config_count = len(ABLATION_CONFIGS)
    for idx, (name, cfg_ab) in enumerate(ABLATION_CONFIGS.items(), 1):
        print(f"\n  [ABLATION {idx}/{config_count}] Config: {name}", flush=True)
        def _make_model(c=cfg_ab):
            return AblationVariant(**c)
        res = quick_train_cv(
            _make_model, X, y,
            n_folds=3, epochs=25,
            lr=CFG['lr'], label_smoothing=0.05
        )
        df = res['fold_df']
        m_acc = df['val_acc'].mean() * 100
        s_acc = df['val_acc'].std()  * 100
        m_f1  = df['f1'].mean()
        m_auc = df['auc_roc'].mean() * 100
        n_params = count_params(AblationVariant(**cfg_ab))
        ablation_rows.append({
            'Configuration': name,
            'Mean Acc (%)': round(m_acc, 2),
            'Std Acc (%)':  round(s_acc, 2),
            'F1-Score':     round(m_f1, 4),
            'AUC-ROC (%)':  round(m_auc, 2),
            '# Params':     f'{n_params:,}',
        })
        all_accs[name] = df['val_acc'].values
        print(f"  [ABLATION {idx}/{config_count}] Result: {m_acc:.2f}% ± {s_acc:.2f}%", flush=True)

    abl_df = pd.DataFrame(ablation_rows)
    print("\n[ABLATION RESULTS]")
    print(abl_df.to_string(index=False))

    csv_path = os.path.join(CFG['results_dir'], f'{tag}_ablation_study.csv')
    abl_df.to_csv(csv_path, index=False)
    print(f"[RESULTS] Saved {csv_path}")

    plot_ablation(abl_df, tag=tag)

    # Statistical tests between full model and others
    full_accs = all_accs['Full Model (Proposed)']
    stat_rows = []
    for name, accs in all_accs.items():
        if name == 'Full Model (Proposed)':
            continue
        if len(accs) == len(full_accs):
            t, p, d = paired_ttest(full_accs, accs)
            stat_rows.append({
                'Pair': f'Full vs {name}',
                't-stat': round(t, 4),
                'p-value': round(p, 5),
                'Sig (p<0.05)': 'Yes' if p < 0.05 else 'No',
                "Cohen's d": round(d, 4),
            })
    stat_df = pd.DataFrame(stat_rows)
    stat_csv = os.path.join(CFG['results_dir'], f'{tag}_ablation_stats.csv')
    stat_df.to_csv(stat_csv, index=False)
    print(f"[RESULTS] Saved {stat_csv}")

    return abl_df, stat_df, all_accs


def run_baseline_comparison(X, y, proposed_accs, tag='full'):
    """Compare proposed model against all baseline models."""
    print("\n" + "="*70)
    print("  BASELINE MODEL COMPARISON (4 Models)")
    print("="*70)

    comparison_rows = []
    all_bl_accs = {}
    model_metrics = {}

    # First add proposed model
    m_acc, ci_acc = compute_95ci(proposed_accs)
    comparison_rows.append({
        'Model': '**Proposed (Ours)**',
        'Mean Acc (%)': round(m_acc * 100, 2),
        '95% CI (±%)': round(ci_acc * 100, 2),
        'Mean AUC (%)': '—',
    })
    model_metrics['Proposed'] = {
        'acc': m_acc, 'prec': 0, 'rec': 0, 'f1': 0, 'auc': 0
    }

    baseline_count = len(BASELINE_MODELS)
    for idx, (name, model_cls) in enumerate(BASELINE_MODELS.items(), 1):
        print(f"\n  [BASELINE {idx}/{baseline_count}] Training: {name}", flush=True)
        def _make_baseline(mc=model_cls):
            return mc()
        res = quick_train_cv(
            _make_baseline, X, y,
            n_folds=3, epochs=20,
            lr=CFG['lr'], label_smoothing=0.05
        )
        df = res['fold_df']
        m_acc_bl, ci_bl = compute_95ci(df['val_acc'].values)
        m_auc_bl = df['auc_roc'].mean()
        m_prec = df['precision'].mean()
        m_rec  = df['recall'].mean()
        m_f1   = df['f1'].mean()
        comparison_rows.append({
            'Model': name,
            'Mean Acc (%)': round(m_acc_bl * 100, 2),
            '95% CI (±%)': round(ci_bl * 100, 2),
            'Mean AUC (%)': round(m_auc_bl * 100, 2),
        })
        all_bl_accs[name] = df['val_acc'].values
        model_metrics[name] = {
            'acc': m_acc_bl, 'prec': m_prec,
            'rec': m_rec, 'f1': m_f1, 'auc': m_auc_bl
        }
        print(f"  [BASELINE {idx}/{baseline_count}] Result: {m_acc_bl*100:.2f}% ± {ci_bl*100:.2f}%", flush=True)

    comp_df = pd.DataFrame(comparison_rows)
    print("\n[BASELINE COMPARISON]")
    print(comp_df.to_string(index=False))

    csv_path = os.path.join(CFG['results_dir'], f'{tag}_baseline_comparison.csv')
    comp_df.to_csv(csv_path, index=False)
    print(f"[RESULTS] Saved {csv_path}")

    plot_baseline_comparison(comp_df, tag=tag)
    plot_per_metric_radar(model_metrics, tag=tag)

    # Statistical significance
    stat_df = full_statistical_report(proposed_accs, all_bl_accs)
    stat_csv = os.path.join(CFG['results_dir'], f'{tag}_statistical_tests.csv')
    stat_df.to_csv(stat_csv, index=False)
    print(f"[RESULTS] Saved {stat_csv}")
    plot_statistical_significance(stat_df, tag=tag)

    return comp_df, stat_df, model_metrics


# ==============================================================================
# 12.  LaTeX TABLE GENERATORS
# ==============================================================================

def df_to_latex(df, caption, label, col_format=None):
    """Convert DataFrame to LaTeX table string."""
    n_cols = len(df.columns)
    if col_format is None:
        col_format = 'l' + 'c' * (n_cols - 1)
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{tab:{label}}}',
        r'\begin{tabular}{' + col_format + '}',
        r'\toprule',
    ]
    # Header
    header = ' & '.join([f'\\textbf{{{c}}}' for c in df.columns]) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    # Rows
    for _, row in df.iterrows():
        lines.append(' & '.join([str(v) for v in row.values]) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def generate_latex_tables(full_res, abl_df, abl_stat_df,
                          comp_df, stat_df, cross_df,
                          lat_mean, lat_std):
    """Generate all LaTeX tables and write to results/."""

    # -- Table 1: CV Results -------------------------------------------------
    fold_df = full_res['fold_df'].copy()
    fold_df['val_acc']   = (fold_df['val_acc'] * 100).round(2)
    fold_df['precision'] = fold_df['precision'].round(4)
    fold_df['recall']    = fold_df['recall'].round(4)
    fold_df['f1']        = fold_df['f1'].round(4)
    fold_df['auc_roc']   = (fold_df['auc_roc'] * 100).round(2)
    fold_df.columns = ['Fold', 'Val Acc (%)', 'Precision', 'Recall',
                       'F1-Score', 'AUC-ROC (%)']
    t1 = df_to_latex(fold_df,
                     'Cross-Validation Results (Proposed Model)',
                     'cv_results')

    # -- Table 2: Ablation ---------------------------------------------------
    t2 = df_to_latex(abl_df,
                     'Ablation Study Results',
                     'ablation')

    # -- Table 3: Ablation statistical tests ---------------------------------
    t3 = df_to_latex(abl_stat_df,
                     'Statistical Analysis: Ablation Configurations (Paired t-test)',
                     'ablation_stats')

    # -- Table 4: Baseline comparison ----------------------------------------
    t4 = df_to_latex(comp_df,
                     'Performance Comparison with Baseline Models',
                     'baseline_comparison')

    # -- Table 5: Statistical significance vs baselines ---------------------
    t5 = df_to_latex(stat_df,
                     'Statistical Significance Testing (Proposed vs Baselines)',
                     'statistical_tests')

    # -- Table 6: Cross-dataset ----------------------------------------------
    t6 = df_to_latex(cross_df,
                     'Cross-Dataset Validation Results',
                     'cross_dataset')

    # -- Write to file -------------------------------------------------------
    tables_path = os.path.join(CFG['results_dir'], 'all_latex_tables.tex')
    with open(tables_path, 'w') as f:
        f.write("% ============================================================\n")
        f.write("% AUTO-GENERATED LaTeX TABLES\n")
        f.write(f"% Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("% Include these in Final_Paper_EEG.tex\n")
        f.write("% ============================================================\n\n")
        for title, tex in [
            ("Table 1: CV Results", t1),
            ("Table 2: Ablation Study", t2),
            ("Table 3: Ablation Statistical Tests", t3),
            ("Table 4: Baseline Comparison", t4),
            ("Table 5: Statistical Significance", t5),
            ("Table 6: Cross-Dataset Validation", t6),
        ]:
            f.write(f"% --- {title} ---\n")
            f.write(tex + "\n\n")

    print(f"[LaTeX] Tables saved to {tables_path}")
    return tables_path


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    t_start = time.time()
    print(f"\n{'='*70}", flush=True)
    print("  COMPLETE EEG CLASSIFICATION PIPELINE", flush=True)
    print("  Enhanced Triple-Layer MetaModel", flush=True)
    print(f"  Device: {DEVICE}  |  Seed: {CFG['seed']}", flush=True)
    print(f"{'='*70}\n", flush=True)
    print("[START] Complete pipeline initialized. Beginning execution...", flush=True)

    # ----------------------------------------------------------------------
    # STEP 1: LOAD FULL DATASET
    # ----------------------------------------------------------------------
    print("\n[PROGRESS] Loading full dataset...", flush=True)
    X_full, y_full = load_dataset(
        max_files=CFG['full_files'],
        train_dir=CFG['train_dir'],
        test_dir=CFG['test_dir'],
        cache_prefix='full'
    )
    print(f"[DATA] Full dataset: {X_full.shape}, Alc={y_full.sum()}, Ctrl={(y_full==0).sum()}")

    # ----------------------------------------------------------------------
    # STEP 2: FULL DATASET TRAINING (10-Fold CV)
    # ----------------------------------------------------------------------
    print("\n[PROGRESS] Starting Full Dataset Training (10-Fold CV)...", flush=True)
    full_res, lat_mean, lat_std = run_full_training(X_full, y_full)
    proposed_accs = full_res['fold_df']['val_acc'].values
    print("[PROGRESS] Full Dataset Training completed.", flush=True)

    # ----------------------------------------------------------------------
    # STEP 3: ABLATION STUDY
    # ----------------------------------------------------------------------
    print("\n[PROGRESS] Starting Ablation Study...", flush=True)
    abl_df, abl_stat_df, ablation_accs = run_ablation_study(X_full, y_full, tag='full')
    print("[PROGRESS] Ablation Study completed.", flush=True)

    # ----------------------------------------------------------------------
    # STEP 4: BASELINE COMPARISON
    # ----------------------------------------------------------------------
    print("\n[PROGRESS] Starting Baseline Comparison...", flush=True)
    comp_df, stat_df, model_metrics = run_baseline_comparison(X_full, y_full, proposed_accs, tag='full')
    print("[PROGRESS] Baseline Comparison completed.", flush=True)

    # ----------------------------------------------------------------------
    # STEP 5: CROSS-DATASET VALIDATION
    # ----------------------------------------------------------------------
    print("\n[PROGRESS] Starting Cross-Dataset Validation...", flush=True)
    n = len(X_full)
    split = int(n * 0.7)
    idx = np.random.permutation(n)
    tr_idx, te_idx = idx[:split], idx[split:]
    cross_results, cross_df = run_cross_dataset(
        X_full[tr_idx], y_full[tr_idx],
        X_full[te_idx], y_full[te_idx],
    )
    print("[PROGRESS] Cross-Dataset Validation completed.", flush=True)

    # ----------------------------------------------------------------------
    # STEP 6: GENERATE LaTeX TABLES
    # ----------------------------------------------------------------------
    print("\n[PROGRESS] Generating LaTeX tables...", flush=True)
    tables_path = generate_latex_tables(
        full_res, abl_df, abl_stat_df,
        comp_df, stat_df, cross_df,
        lat_mean, lat_std,
    )
    print("[PROGRESS] LaTeX tables generated.", flush=True)

    # ----------------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------------
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*70}", flush=True)
    print(f"[COMPLETE] All experiments finished in {elapsed:.1f} minutes", flush=True)
    print(f"{'='*70}", flush=True)
    
    df_f = full_res['fold_df']
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"  Accuracy  : {df_f['val_acc'].mean()*100:.2f}% ± {df_f['val_acc'].std()*100:.2f}%")
    print(f"  AUC-ROC   : {df_f['auc_roc'].mean()*100:.2f}% ± {df_f['auc_roc'].std()*100:.2f}%")
    print(f"  F1-Score  : {df_f['f1'].mean():.4f}")
    print(f"  Latency   : {lat_mean:.2f} ± {lat_std:.2f} ms/sample")
    print(f"  Time      : {elapsed:.1f} min")
    print(f"\n  All plots  -> {CFG['plots_dir']}/")
    print(f"  All tables -> {CFG['results_dir']}/")
    print(f"  Models     -> {CFG['models_dir']}/")
    print("="*70)


if __name__ == '__main__':
    main()