import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bio_sts_mamba import get_10_20_spherical_coords

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
    """
    Bio-Topological Dynamic Graph Attention over 64 electrode channels.
    """
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
        # x: (B, 64, 256)
        B, N, T = x.shape
        Q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        alpha = torch.sigmoid(self.blend_logit)
        A_hyb = alpha * self.A_phys.unsqueeze(0).unsqueeze(0) + (1.0 - alpha) * attn

        out = torch.matmul(A_hyb, V).transpose(1, 2).contiguous().view(B, N, -1)
        out = self.norm(self.proj(out))
        return self.act(out) # (B, 64, out_dim)


class BioSTG_MetaNet(nn.Module):
    """
    Novel Unified Architecture:
    1. Spatial Branch: Bio-Topological Scalp Graph Attention (Bio-DGAT)
    2. Spatio-Temporal Branch: Residual CNN Stack + BiLSTM + 8-Head Multi-Head Attention
    3. Spectral Branch: In-Network Tri-Band Parallel Streams (slow delta/theta, mid alpha/beta, fast beta/gamma)
    4. Statistical Branch: Per-channel 6-moment statistics
    5. Meta-Classifier: Highway Gated Fusion
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Branch 1: Bio-Topological Spatial Graph Attention
        self.bio_gat = BioDGAT_SpatialModule(num_nodes=64, in_dim=256, out_dim=64)
        self.spatial_pool = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # Branch 2: Spatio-Temporal CNN + BiLSTM + MHA
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

        # Branch 3: Frequency-Segregated Multi-Scale Streams
        def stream(k):
            return nn.Sequential(
                nn.Conv1d(64, 64, k, padding=k//2, bias=False),
                nn.BatchNorm1d(64), nn.GELU(),
                ResidualBlock(64, 128, k=3, dropout=0.15),
                nn.AdaptiveAvgPool1d(16)
            )
        self.slow = stream(31) # delta/theta
        self.mid  = stream(11) # alpha/beta
        self.fast = stream(3)  # beta/gamma
        self.spec_fuse = nn.Sequential(
            nn.Linear(128 * 16 * 3, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 192), nn.GELU(), nn.Dropout(0.1)
        )

        # Branch 4: Statistical Moments
        self.stat_proj = nn.Sequential(
            nn.Linear(64 * 6, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU()
        )

        # Meta-Classifier Head (Total Dim: 128 + 192 + 192 + 64 = 576)
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
        # 1. Spatial Graph
        f_spat = self.spatial_pool(self.bio_gat(x))

        # 2. Spatio-Temporal
        h = self.cnn(x).permute(0, 2, 1) # (B, 64, 256)
        h_lstm, _ = self.lstm(h)
        h_attn, _ = self.attn(h_lstm, h_lstm, h_lstm)
        h_st = self.norm_st(h_lstm + h_attn).mean(dim=1)
        f_st = self.st_proj(h_st)

        # 3. Frequency Segregation
        s = self.slow(x).reshape(x.size(0), -1)
        m = self.mid(x).reshape(x.size(0), -1)
        f = self.fast(x).reshape(x.size(0), -1)
        f_spec = self.spec_fuse(torch.cat([s, m, f], dim=1))

        # 4. Statistical Features
        f_stat = self.extract_stats(x)

        # 5. Gated Meta-Classification
        fused = torch.cat([f_spat, f_st, f_spec, f_stat], dim=1)
        gated = fused * self.gate(fused)
        return self.classifier(gated)


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    model = BioSTG_MetaNet().to(DEVICE)
    p = sum(p.numel() for p in model.parameters())
    print(f"BioSTG_MetaNet total parameters: {p:,}")

    X = np.load('results/full_X.npy')
    y = np.load('results/full_y.npy')
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X, y))

    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    print("Training 15 epochs on Fold 1...")
    for epoch in range(1, 16):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()

        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                preds = model(bx).argmax(dim=1)
                cor += (preds == by).sum().item()
                tot += len(by)
        print(f"Epoch {epoch:02d}: Val Acc = {cor/tot*100:.2f}%")
