import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. EEGNet (Lawhern et al., 2018 - BCI Gold Standard)
# ==============================================================================

class EEGNet(nn.Module):
    def __init__(self, num_channels=64, num_classes=2, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.25):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: Temporal convolution followed by Depthwise spatial convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable convolution
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * 8, num_classes) # For 256 time steps: 256/4/8 = 8
        )

    def forward(self, x):
        # x: (B, 64, 256) -> expand to (B, 1, 64, 256)
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
        x = self.drop2(self.pool2(self.act2(self.bn3(self.separable(x)))))
        return self.classifier(x)


# ==============================================================================
# 2. ShallowFBCSPNet (Schirrmeister et al., 2017)
# ==============================================================================

class ShallowFBCSPNet(nn.Module):
    def __init__(self, num_channels=64, num_classes=2, n_filters_time=40, filter_time_length=25):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=False)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_time, (num_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_time)
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters_time * 29, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        # Log square activation for energy / band-power estimation
        x = torch.log(torch.clamp(self.pool(x ** 2), min=1e-6))
        x = self.drop(x)
        return self.fc(x)


# ==============================================================================
# 3. DeepConvNet (Schirrmeister et al., 2017)
# ==============================================================================

class DeepConvNet(nn.Module):
    def __init__(self, num_channels=64, num_classes=2):
        super().__init__()
        self.conv_time = nn.Conv2d(1, 25, (1, 10), bias=False)
        self.conv_spat = nn.Conv2d(25, 25, (num_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d((1, 3), stride=(1, 3))

        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 10), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 10), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(100 * 5, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.pool1(F.elu(self.bn1(x)))
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ==============================================================================
# 4. CNN-LSTM Hybrid (Li & Wu, 2022)
# ==============================================================================

class CNNLSTM_Hybrid(nn.Module):
    def __init__(self, num_channels=64, num_classes=2):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 64, 256)
        c = self.conv_stack(x) # (B, 256, 64)
        c = c.permute(0, 2, 1) # (B, 64, 256)
        out, _ = self.lstm(c)  # (B, 64, 256)
        out = out.mean(dim=1)  # (B, 256)
        return self.fc(out)


# ==============================================================================
# 5. PCA + ANN (Farsi et al., 2021)
# ==============================================================================

class PCA_ANN(nn.Module):
    def __init__(self, in_features=64*256, pca_dim=128, num_classes=2):
        super().__init__()
        # Linear projection mimicking PCA dimensionality reduction
        self.proj = nn.Linear(in_features, pca_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pca_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        flat = x.view(x.shape[0], -1)
        z = self.proj(flat)
        return self.mlp(z)


# ==============================================================================
# 6. Vanilla BiLSTM & Vanilla CNN
# ==============================================================================

class VanillaBiLSTM(nn.Module):
    def __init__(self, num_channels=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (B, 64, 256) -> permute to (B, 256, 64)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out.mean(dim=1))


class VanillaCNN(nn.Module):
    def __init__(self, num_channels=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# 7. EEG-Conformer (Song et al., 2023 - Hybrid Conv-Transformer)
# ==============================================================================

class EEGConformer(nn.Module):
    def __init__(self, num_channels=64, num_classes=2, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d_model, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.Conv2d(d_model, d_model, (num_channels, 1)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, 4)) # Time downsampled to 64
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=128, dropout=0.2,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (B, 64, 256) -> (B, 1, 64, 256)
        x = x.unsqueeze(1)
        feat = self.patch_embed(x) # (B, d_model, 1, 64)
        seq = feat.squeeze(2).permute(0, 2, 1) # (B, 64, d_model)
        trans_out = self.transformer(seq) # (B, 64, d_model)
        pool = trans_out.mean(dim=1)      # (B, d_model)
        return self.classifier(pool)


def get_all_baselines(num_channels=64, num_classes=2):
    """Factory dictionary of all benchmark models."""
    return {
        'EEGNet': EEGNet(num_channels=num_channels, num_classes=num_classes),
        'ShallowFBCSPNet': ShallowFBCSPNet(num_channels=num_channels, num_classes=num_classes),
        'DeepConvNet': DeepConvNet(num_channels=num_channels, num_classes=num_classes),
        'EEGConformer': EEGConformer(num_channels=num_channels, num_classes=num_classes),
        'CNNLSTM_Hybrid': CNNLSTM_Hybrid(num_channels=num_channels, num_classes=num_classes),
        'VanillaCNN': VanillaCNN(num_channels=num_channels, num_classes=num_classes),
        'VanillaBiLSTM': VanillaBiLSTM(num_channels=num_channels, num_classes=num_classes),
        'PCA_ANN': PCA_ANN(in_features=num_channels*256, num_classes=num_classes)
    }


if __name__ == '__main__':
    dummy = torch.randn(2, 64, 256)
    models = get_all_baselines()
    print("Testing forward passes for all 8 baseline architectures:")
    for name, m in models.items():
        out = m(dummy)
        p = sum(p.numel() for p in m.parameters())
        print(f"  [{name:16s}] Output: {list(out.shape)}  Params: {p:,}")
    print("[SUCCESS] All baselines operational!")
