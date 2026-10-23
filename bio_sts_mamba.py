import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. 10-20 STANDARD 3D ELECTRODE TOPOLOGY
# ==============================================================================

# Standard 64-channel list sorted alphabetically as parsed from SMNI EEG
SMNI_64_CHANNELS = [
    'AF1', 'AF2', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ', 'CZ', 'F1', 'F2', 'F3',
    'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
    'FCZ', 'FP1', 'FP2', 'FPZ', 'FT7', 'FT8', 'FZ', 'O1', 'O2', 'OZ', 'P1',
    'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO1', 'PO2', 'PO7', 'PO8',
    'POZ', 'PZ', 'T7', 'T8', 'TP7', 'TP8', 'X', 'Y', 'nd'
]

def get_10_20_spherical_coords():
    """
    Approximate 3D coordinates (x: Left-Right, y: Posterior-Anterior, z: Inferior-Superior)
    on a normalized unit sphere for 10-20 montage electrodes.
    """
    coords_dict = {
        'FP1': (-0.31, 0.95, 0.0),  'FPZ': (0.0, 0.99, 0.1),   'FP2': (0.31, 0.95, 0.0),
        'AF7': (-0.59, 0.78, 0.18), 'AF1': (-0.18, 0.88, 0.44), 'AFZ': (0.0, 0.89, 0.45),
        'AF2': (0.18, 0.88, 0.44),  'AF8': (0.59, 0.78, 0.18),
        'F7':  (-0.81, 0.54, 0.22), 'F5':  (-0.64, 0.57, 0.51), 'F3':  (-0.43, 0.63, 0.65),
        'F1':  (-0.19, 0.64, 0.74), 'FZ':  (0.0, 0.65, 0.76),   'F2':  (0.19, 0.64, 0.74),
        'F4':  (0.43, 0.63, 0.65),  'F6':  (0.64, 0.57, 0.51),  'F8':  (0.81, 0.54, 0.22),
        'FT7': (-0.91, 0.30, 0.28), 'FC5': (-0.73, 0.33, 0.59), 'FC3': (-0.49, 0.36, 0.79),
        'FC1': (-0.21, 0.36, 0.91), 'FCZ': (0.0, 0.37, 0.93),   'FC2': (0.21, 0.36, 0.91),
        'FC4': (0.49, 0.36, 0.79),  'FC6': (0.73, 0.33, 0.59),  'FT8': (0.91, 0.30, 0.28),
        'T7':  (-0.95, 0.0, 0.31),  'C5':  (-0.76, 0.0, 0.65),  'C3':  (-0.51, 0.0, 0.86),
        'C1':  (-0.21, 0.0, 0.98),  'CZ':  (0.0, 0.0, 1.0),     'C2':  (0.21, 0.0, 0.98),
        'C4':  (0.51, 0.0, 0.86),   'C6':  (0.76, 0.0, 0.65),   'T8':  (0.95, 0.0, 0.31),
        'TP7': (-0.91, -0.30, 0.28),'CP5': (-0.73, -0.33, 0.59),'CP3': (-0.49, -0.36, 0.79),
        'CP1': (-0.21, -0.36, 0.91),'CPZ': (0.0, -0.37, 0.93),  'CP2': (0.21, -0.36, 0.91),
        'CP4': (0.49, -0.36, 0.79), 'CP6': (0.73, -0.33, 0.59), 'TP8': (0.91, -0.30, 0.28),
        'P7':  (-0.81, -0.54, 0.22),'P5':  (-0.64, -0.57, 0.51),'P3':  (-0.43, -0.63, 0.65),
        'P1':  (-0.19, -0.64, 0.74),'PZ':  (0.0, -0.65, 0.76),  'P2':  (0.19, -0.64, 0.74),
        'P4':  (0.43, -0.63, 0.65), 'P6':  (0.64, -0.57, 0.51), 'P8':  (0.81, -0.54, 0.22),
        'PO7': (-0.59, -0.78, 0.18),'PO1': (-0.18, -0.88, 0.44),'POZ': (0.0, -0.89, 0.45),
        'PO2': (0.18, -0.88, 0.44), 'PO8': (0.59, -0.78, 0.18),
        'O1':  (-0.31, -0.95, 0.0), 'OZ':  (0.0, -0.99, 0.1),   'O2':  (0.31, -0.95, 0.0),
        'X':   (-0.5, 0.0, -0.5),   'Y':   (0.5, 0.0, -0.5),    'nd':  (0.0, 0.0, -0.5)
    }
    coords = np.zeros((len(SMNI_64_CHANNELS), 3), dtype=np.float32)
    for i, ch in enumerate(SMNI_64_CHANNELS):
        if ch in coords_dict:
            coords[i] = coords_dict[ch]
        else:
            coords[i] = [0.0, 0.0, 0.0]
    return coords


# ==============================================================================
# 2. LEARNABLE PARAMETERIZED SPECTRAL FILTERBANK (Bio-SincNet)
# ==============================================================================

class BioSincConv1d(nn.Module):
    """
    Learnable parameterized bandpass filterbank based on Sinc functions.
    Initializes bandpass filters across standard EEG rhythms:
    Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz).
    Filter cutoffs are learnable parameters optimized via backpropagation.
    """
    def __init__(self, out_channels=16, kernel_size=61, sample_rate=256, min_low_hz=0.5, min_band_hz=2.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize band frequencies biologically
        # Distribute across standard neurological bands
        n_bands = out_channels
        f_init = []
        band_ranges = [
            (0.5, 4.0),   # Delta
            (4.0, 8.0),   # Theta
            (8.0, 13.0),  # Alpha
            (13.0, 30.0), # Beta
            (30.0, 45.0)  # Gamma
        ]
        filters_per_range = n_bands // len(band_ranges)
        remainder = n_bands % len(band_ranges)

        for idx, (lo, hi) in enumerate(band_ranges):
            count = filters_per_range + (1 if idx < remainder else 0)
            los = np.linspace(lo, hi - 1.0, count)
            widths = np.full(count, (hi - lo) / count)
            for l, w in zip(los, widths):
                f_init.append((l, max(w, min_band_hz)))

        f_init = np.array(f_init, dtype=np.float32)

        # Learnable parameters (normalized to Nyquist frequency = sample_rate / 2)
        nyq = sample_rate / 2.0
        self.f_low = nn.Parameter(torch.tensor(f_init[:, 0] / nyq).view(-1, 1))
        self.f_band = nn.Parameter(torch.tensor(f_init[:, 1] / nyq).view(-1, 1))

        # Half-length time index
        n = torch.linspace(0, (kernel_size - 1) / 2, steps=int((kernel_size + 1) / 2))
        self.register_buffer('n_', 2 * math.pi * n / sample_rate)

        # Hamming window buffer
        t = torch.linspace(0, kernel_size - 1, steps=kernel_size)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * t / (kernel_size - 1))
        self.register_buffer('window', window)

    def forward(self, x):
        """
        x: (Batch, In_Channels=64, Time=256)
        Returns: (Batch, In_Channels=64, Out_Channels=16, Time=256)
        """
        min_lo = self.min_low_hz / (self.sample_rate / 2.0)
        min_bd = self.min_band_hz / (self.sample_rate / 2.0)
        f_low_clamped = torch.clamp(self.f_low, min=min_lo, max=0.90)
        f_band_clamped = torch.clamp(self.f_band, min=min_bd, max=0.90)
        f_high = torch.clamp(f_low_clamped + f_band_clamped, min=min_lo + min_bd, max=0.99)

        # Scaled Sinc functions
        f_high_t = 2 * math.pi * (f_high * (self.sample_rate / 2.0))
        f_low_t = 2 * math.pi * (f_low_clamped * (self.sample_rate / 2.0))

        # Compute sinc filters for n > 0
        n_scaled = self.n_  # (kernel_size + 1)/2
        band_high = torch.sin(f_high_t * n_scaled) / (math.pi * n_scaled + 1e-12)
        band_low = torch.sin(f_low_t * n_scaled) / (math.pi * n_scaled + 1e-12)
        band_pass_half = band_high - band_low

        # n=0 entry
        band_pass_0 = 2 * (f_high - f_low_clamped) * (self.sample_rate / 2.0) / self.sample_rate

        # Symmetrize filter
        band_pass_left = torch.flip(band_pass_half[:, 1:], dims=[-1])
        filters = torch.cat([band_pass_left, band_pass_0, band_pass_half[:, 1:]], dim=-1) # (Out_Channels, Kernel_Size)

        # Apply Hamming window
        filters = filters * self.window.unsqueeze(0)

        # Normalize filter energy
        filters = filters / (torch.norm(filters, dim=-1, keepdim=True) + 1e-8)

        # Reshape for conv1d: (Out_Channels, 1, Kernel_Size)
        filters = filters.unsqueeze(1)

        B, C, T = x.shape
        x_reshaped = x.view(B * C, 1, T)
        padding = (self.kernel_size - 1) // 2
        out = F.conv1d(x_reshaped, filters, padding=padding) # (B*C, Out_Channels, T)
        out = out.view(B, C, self.out_channels, T)
        return out


# ==============================================================================
# 3. BIO-TOPOLOGICAL DYNAMIC GRAPH ATTENTION (Bio-DGAT)
# ==============================================================================

class BioDGATLayer(nn.Module):
    """
    Bio-Topological Dynamic Graph Attention Network:
    Blends physical scalp electrode proximity (derived from standard 10-20 spherical coordinates)
    with dynamic functional cross-channel attention.
    """
    def __init__(self, in_features, out_features, num_nodes=64, num_heads=4, sigma=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads

        coords = get_10_20_spherical_coords() # (64, 3)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :] # (64, 64, 3)
        dist_sq = np.sum(diff ** 2, axis=-1) # (64, 64)
        A_phys = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
        np.fill_diagonal(A_phys, 1.0)
        deg = np.sum(A_phys, axis=1)
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        D_mat = np.diag(deg_inv_sqrt)
        A_norm = D_mat.dot(A_phys).dot(D_mat)

        self.register_buffer('A_phys', torch.tensor(A_norm, dtype=torch.float32))

        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: (Batch, Num_Nodes=64, In_Features)
        Returns: (Batch, Num_Nodes=64, Out_Features)
        """
        B, N, F_in = x.shape
        residual = x

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        A_dyn = F.softmax(scores, dim=-1)

        alpha = torch.sigmoid(self.blend_logit)
        A_phys_exp = self.A_phys.unsqueeze(0).unsqueeze(0)
        A_hybrid = alpha * A_phys_exp + (1.0 - alpha) * A_dyn

        out = torch.matmul(A_hybrid, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_features)
        out = self.out_proj(out)

        if F_in == self.out_features:
            out = self.norm(out + residual)
        else:
            out = self.norm(out)
        return self.act(out)


# ==============================================================================
# 4. BI-DIRECTIONAL SELECTIVE STATE-SPACE SEQUENCE MODULE (Bi-Mamba)
# ==============================================================================

class BiMambaBlock(nn.Module):
    """
    Hardware-efficient Pure-PyTorch Bi-directional Selective State-Space Sequence Model (SSM).
    Captures long-range temporal phase alignments with linear-time O(T) complexity.
    """
    def __init__(self, d_model=128, d_state=16, dt_rank=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2)

        self.A_log_fwd = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.A_log_bwd = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))

        self.x_proj_fwd = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj_fwd = nn.Linear(dt_rank, d_model, bias=True)

        self.x_proj_bwd = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj_bwd = nn.Linear(dt_rank, d_model, bias=True)

        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

    def _selective_scan(self, u, A_log, x_proj, dt_proj):
        B, T, D = u.shape
        A = -torch.exp(A_log)

        x_dbl = x_proj(u)
        dt_rank = dt_proj.in_features
        delta = x_dbl[..., :dt_rank]
        B_mat = x_dbl[..., dt_rank:dt_rank + self.d_state]
        C_mat = x_dbl[..., dt_rank + self.d_state:]

        delta = F.softplus(dt_proj(delta))

        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)

        h = torch.zeros(B, D, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            y = torch.sum(h * C_mat[:, t].unsqueeze(1), dim=-1)
            ys.append(y)
        return torch.stack(ys, dim=1)

    def forward(self, x):
        residual = x
        x_proj = self.in_proj(x)
        u, gate = x_proj.chunk(2, dim=-1)
        u = self.act(u)

        y_fwd = self._selective_scan(u, self.A_log_fwd, self.x_proj_fwd, self.dt_proj_fwd)

        u_flip = torch.flip(u, dims=[1])
        y_bwd_flip = self._selective_scan(u_flip, self.A_log_bwd, self.x_proj_bwd, self.dt_proj_bwd)
        y_bwd = torch.flip(y_bwd_flip, dims=[1])

        y_cat = torch.cat([y_fwd, y_bwd], dim=-1)
        out = self.out_proj(y_cat * torch.sigmoid(torch.cat([gate, gate], dim=-1)))
        return self.norm(out + residual)


# ==============================================================================
# 5. SUBJECT-INVARIANT SUPERVISED CONTRASTIVE LOSS
# ==============================================================================

class SubjectInvariantSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with Subject De-biasing.
    Maximizes agreement between representations of the same diagnostic class
    across different subjects, penalizing subject-specific memorization.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, subjects=None):
        features = F.normalize(features, p=2, dim=1)
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        sim_matrix = torch.matmul(features, features.T) / self.temperature

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag_mask = torch.eye(B, device=features.device, dtype=torch.bool)
        labels_eq = labels_eq & ~diag_mask

        if subjects is not None:
            subj_diff = torch.tensor([[s1 != s2 for s2 in subjects] for s1 in subjects], device=features.device)
            cross_subj_pos = labels_eq & subj_diff
            if cross_subj_pos.sum() > 0:
                pos_mask = cross_subj_pos
            else:
                pos_mask = labels_eq
        else:
            pos_mask = labels_eq

        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        exp_logits = torch.exp(logits) * (~diag_mask).float()
        denom = exp_logits.sum(dim=1, keepdim=True) + 1e-8

        log_prob = logits - torch.log(denom)

        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1)[valid] / pos_counts[valid]
        loss = -mean_log_prob.mean()
        return loss


# ==============================================================================
# 6. UNIFIED MODEL ARCHITECTURE: Bio-STS-Mamba
# ==============================================================================

class BioSTSMamba(nn.Module):
    """
    Bio-Topological Spatio-Temporal-Spectral Mamba Graph Network.
    Integrates:
    - BioSincNet (Learnable spectral filtering across delta, theta, alpha, beta, gamma)
    - BioDGAT (Topological graph attention using 10-20 spherical electrode geometry)
    - BiMamba (Bi-directional selective state-space sequence modeling)
    - Statistical feature fusion & Squeeze-Excitation meta-gating
    """
    def __init__(self, num_channels=64, time_steps=256, num_classes=2,
                 sinc_filters=16, sinc_kernel=61,
                 d_model=128, num_gat_heads=4, mamba_state=16):
        super().__init__()
        self.num_channels = num_channels
        self.time_steps = time_steps

        # 1. Spectral Layer: Learnable SincNet
        self.sinc_net = BioSincConv1d(out_channels=sinc_filters, kernel_size=sinc_kernel, sample_rate=256)
        
        self.spectral_compress = nn.Sequential(
            nn.Conv2d(sinc_filters, d_model // 2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((None, 32))
        )

        # 2. Spatial Layer: Bio-Topological Graph Attention
        self.bio_gat = BioDGATLayer(in_features=d_model, out_features=d_model,
                                    num_nodes=num_channels, num_heads=num_gat_heads)

        # 3. Temporal Layer: Bi-directional Mamba Sequence Block
        self.bi_mamba = BiMambaBlock(d_model=d_model, d_state=mamba_state)

        # 4. Statistical Moments Branch (Mean, Std, Max, Min, Skew, Kurtosis)
        self.stat_mlp = nn.Sequential(
            nn.Linear(num_channels * 6, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        # 5. Projection Head for Contrastive Regularization
        self.proj_head = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

        # 6. Gated Meta-Classifier Head
        combined_dim = d_model + 64
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def extract_statistical_features(self, x):
        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1) + 1e-8
        max_val, _ = torch.max(x, dim=-1)
        min_val, _ = torch.min(x, dim=-1)
        diff = x - mean.unsqueeze(-1)
        skew = torch.mean(diff ** 3, dim=-1) / (std ** 3)
        kurt = torch.mean(diff ** 4, dim=-1) / (std ** 4)
        stat = torch.cat([mean, std, max_val, min_val, skew, kurt], dim=-1)
        return self.stat_mlp(stat)

    def forward(self, x, return_embedding=False):
        B, C, T = x.shape

        # Step 1: Spectral Bandpass Decomposition
        x_sinc = self.sinc_net(x)                      # (B, C=64, Filters=16, T=256)
        x_sinc = x_sinc.permute(0, 2, 1, 3)            # (B, Filters=16, C=64, T=256)
        x_spec = self.spectral_compress(x_sinc)        # (B, d_model=128, C=64, Time=32)

        # Step 2: Spatial Topological Graph Attention
        x_nodes = x_spec.mean(dim=-1).permute(0, 2, 1) # (B, 64, d_model)
        x_spatial = self.bio_gat(x_nodes)              # (B, 64, d_model)

        # Step 3: Temporal Bi-Mamba Sequence Modeling
        x_temporal_seq = x_spec.mean(dim=2).permute(0, 2, 1) # (B, Time=32, d_model)
        x_temporal = self.bi_mamba(x_temporal_seq)           # (B, Time=32, d_model)
        x_temporal_pool = x_temporal.mean(dim=1)             # (B, d_model)

        x_spatial_pool = x_spatial.mean(dim=1)               # (B, d_model)

        h_st = (x_spatial_pool + x_temporal_pool) / 2.0      # (B, d_model)
        h_stat = self.extract_statistical_features(x)        # (B, 64)

        h_all = torch.cat([h_st, h_stat], dim=-1)            # (B, d_model + 64)

        z_proj = self.proj_head(h_all)                       # (B, 64)

        gated_h = h_all * self.gate(h_all)
        logits = self.classifier(gated_h)

        if return_embedding:
            return logits, z_proj
        return logits


if __name__ == '__main__':
    model = BioSTSMamba()
    dummy_x = torch.randn(4, 64, 256)
    logits, z = model(dummy_x, return_embedding=True)
    print(f"[TEST SUCCESS] BioSTSMamba output logits: {logits.shape}, z: {z.shape}")
    loss_fn = SubjectInvariantSupConLoss()
    dummy_labels = torch.tensor([1, 1, 0, 0])
    dummy_subjects = ['s1', 's2', 's3', 's4']
    loss = loss_fn(z, dummy_labels, dummy_subjects)
    loss.backward()
    print(f"[TEST SUCCESS] Contrastive loss: {loss.item():.4f}, backward pass successful.")
