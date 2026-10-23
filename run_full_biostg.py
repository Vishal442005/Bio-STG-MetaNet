import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)

from bio_sts_mamba import get_10_20_spherical_coords

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[ENVIRONMENT] Active Device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Publication plot styling
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
    'grid.alpha': 0.25,
    'grid.linestyle': '--'
})
sns.set_theme(style='ticks', context='paper', font_scale=1.2)


# ==============================================================================
# MODEL ARCHITECTURE: Bio-STG-MetaNet
# ==============================================================================

class SEBlock(nn.Module):
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
        return x * self.fc(x).unsqueeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k=3, dropout=0.15):
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


class BioDGAT_SpatialModule(nn.Module):
    """Bio-Topological Dynamic Graph Attention over 64 electrode channels."""
    def __init__(self, num_nodes=64, in_dim=256, out_dim=64, num_heads=4, sigma=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        coords = get_10_20_spherical_coords()
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        A_phys = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
        np.fill_diagonal(A_phys, 1.0)
        deg = np.sum(A_phys, axis=1)
        deg_inv = np.power(deg, -0.5)
        deg_inv[np.isinf(deg_inv)] = 0.0
        D = np.diag(deg_inv)
        A_norm = D.dot(A_phys).dot(D)
        self.register_buffer('A_phys', torch.tensor(A_norm, dtype=torch.float32))

        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.blend_logit = nn.Parameter(torch.tensor(0.0))
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, N, T = x.shape
        Q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        alpha = torch.sigmoid(self.blend_logit)
        A_hyb = alpha * self.A_phys.unsqueeze(0).unsqueeze(0) + (1.0 - alpha) * attn

        out = torch.matmul(A_hyb, V).transpose(1, 2).contiguous().view(B, N, -1)
        out = self.norm(self.proj(out))
        return self.act(out)


class BioSTG_MetaNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 1. Spatial Graph
        self.bio_gat = BioDGAT_SpatialModule(num_nodes=64, in_dim=256, out_dim=64)
        self.spatial_pool = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # 2. Spatio-Temporal CNN + BiLSTM + MHA
        self.cnn = nn.Sequential(
            ResidualBlock(64, 128, k=7, dropout=0.15),
            nn.MaxPool1d(2),
            ResidualBlock(128, 256, k=5, dropout=0.15),
            nn.MaxPool1d(2),
            ResidualBlock(256, 256, k=3, dropout=0.1)
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1, batch_first=True)
        self.norm_st = nn.LayerNorm(256)
        self.st_proj = nn.Sequential(nn.Linear(256, 192), nn.GELU(), nn.Dropout(0.15))

        # 3. Frequency Segregation (slow delta/theta, mid alpha/beta, fast beta/gamma)
        def stream(k):
            return nn.Sequential(
                nn.Conv1d(64, 64, k, padding=k//2, bias=False),
                nn.BatchNorm1d(64), nn.GELU(),
                ResidualBlock(64, 128, k=3, dropout=0.15),
                nn.AdaptiveAvgPool1d(16)
            )
        self.slow = stream(31)
        self.mid  = stream(11)
        self.fast = stream(3)
        self.spec_fuse = nn.Sequential(
            nn.Linear(128 * 16 * 3, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 192), nn.GELU(), nn.Dropout(0.1)
        )

        # 4. Statistical Moments
        self.stat_proj = nn.Sequential(
            nn.Linear(64 * 6, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU()
        )

        # 5. Gated Meta-Classifier Head
        total_dim = 128 + 192 + 192 + 64
        self.gate = nn.Sequential(nn.Linear(total_dim, total_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def extract_stats(self, x):
        mu = x.mean(-1)
        sd = x.std(-1) + 1e-8
        mx = x.max(-1).values
        mn = x.min(-1).values
        d = x - mu.unsqueeze(-1)
        sk = ((d / sd.unsqueeze(-1)) ** 3).mean(-1)
        kt = ((d / sd.unsqueeze(-1)) ** 4).mean(-1)
        feats = torch.cat([mu, sd, mx, mn, sk, kt], dim=-1)
        return self.stat_proj(feats)

    def forward(self, x):
        f_spat = self.spatial_pool(self.bio_gat(x))

        h = self.cnn(x).permute(0, 2, 1)
        h_lstm, _ = self.lstm(h)
        h_attn, _ = self.attn(h_lstm, h_lstm, h_lstm)
        h_st = self.norm_st(h_lstm + h_attn).mean(dim=1)
        f_st = self.st_proj(h_st)

        s = self.slow(x).reshape(x.size(0), -1)
        m = self.mid(x).reshape(x.size(0), -1)
        f = self.fast(x).reshape(x.size(0), -1)
        f_spec = self.spec_fuse(torch.cat([s, m, f], dim=1))

        f_stat = self.extract_stats(x)

        fused = torch.cat([f_spat, f_st, f_spec, f_stat], dim=1)
        gated = fused * self.gate(fused)
        return self.classifier(gated)


def main():
    print("="*70)
    print("  EVALUATING 10-FOLD CV ON NOVEL Bio-STG-MetaNet")
    print("="*70)
    X = np.load('results/full_X.npy')
    y = np.load('results/full_y.npy')
    print(f"Data: {X.shape}, Alc={y.sum()}, Ctrl={(y==0).sum()}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []
    all_true, all_pred, all_prob = [], [], []
    histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- [Fold {fold}/10] ---")
        train_ds = TensorDataset(torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        model = BioSTG_MetaNet().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_metrics = None
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for ep in range(1, 26):
            model.train()
            tr_loss, tr_cor, tr_tot = 0.0, 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(by)
                preds = out.argmax(dim=1)
                tr_cor += (preds == by).sum().item()
                tr_tot += len(by)

            scheduler.step()

            model.eval()
            val_loss = 0.0
            v_preds, v_probs, v_targets = [], [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    out = model(bx)
                    loss = criterion(out, by)
                    val_loss += loss.item() * len(by)
                    probs = F.softmax(out, dim=1)[:, 1]
                    preds = out.argmax(dim=1)
                    v_probs.extend(probs.cpu().numpy())
                    v_preds.extend(preds.cpu().numpy())
                    v_targets.extend(by.cpu().numpy())

            v_preds = np.array(v_preds)
            v_probs = np.array(v_probs)
            v_targets = np.array(v_targets)
            acc = accuracy_score(v_targets, v_preds)
            auc_v = roc_auc_score(v_targets, v_probs)
            f1_v = f1_score(v_targets, v_preds)

            history['train_loss'].append(tr_loss / tr_tot)
            history['val_loss'].append(val_loss / len(v_targets))
            history['train_acc'].append(tr_cor / tr_tot)
            history['val_acc'].append(acc)

            if acc >= best_val_acc:
                best_val_acc = acc
                best_metrics = {
                    'acc': acc, 'auc': auc_v, 'f1': f1_v,
                    'prec': precision_score(v_targets, v_preds),
                    'rec': recall_score(v_targets, v_preds),
                    'preds': v_preds, 'probs': v_probs, 'targets': v_targets
                }
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'biostg_fold{fold}.pt'))

            if ep % 5 == 0 or ep == 25:
                print(f"  Epoch {ep:02d}/25 | Train Acc: {tr_cor/tr_tot*100:6.2f}% | Val Acc: {acc*100:6.2f}% | Best: {best_val_acc*100:6.2f}%")

        fold_metrics.append({
            'fold': fold,
            'val_acc': best_metrics['acc'],
            'precision': best_metrics['prec'],
            'recall': best_metrics['rec'],
            'f1': best_metrics['f1'],
            'auc_roc': best_metrics['auc']
        })
        all_true.extend(best_metrics['targets'])
        all_pred.extend(best_metrics['preds'])
        all_prob.extend(best_metrics['probs'])
        histories.append(history)

    df_cv = pd.DataFrame(fold_metrics)
    print("\n" + "="*70)
    print("  FINAL 10-FOLD CV RESULTS (Bio-STG-MetaNet)")
    print("="*70)
    print(df_cv.to_string(index=False))
    m_acc, s_acc = df_cv['val_acc'].mean()*100, df_cv['val_acc'].std()*100
    m_auc, s_auc = df_cv['auc_roc'].mean()*100, df_cv['auc_roc'].std()*100
    m_f1 = df_cv['f1'].mean()
    print(f"\nMean Accuracy: {m_acc:.2f}% ± {s_acc:.2f}%")
    print(f"Mean AUC-ROC:  {m_auc:.2f}% ± {s_auc:.2f}%")
    print(f"Mean F1-Score: {m_f1:.4f}")

    df_cv.to_csv(os.path.join(RESULTS_DIR, 'biostg_cv_results.csv'), index=False)
    np.save(os.path.join(RESULTS_DIR, 'biostg_true.npy'), np.array(all_true))
    np.save(os.path.join(RESULTS_DIR, 'biostg_pred.npy'), np.array(all_pred))
    np.save(os.path.join(RESULTS_DIR, 'biostg_prob.npy'), np.array(all_prob))


if __name__ == '__main__':
    main()
