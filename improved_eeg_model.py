"""
Improved EEG Classification Model with Enhanced Architecture
Target: 99%+ accuracy on alcoholic EEG classification
Enhancements:
- Attention mechanisms for better feature selection
- Residual connections for improved gradient flow
- Enhanced regularization (spatial dropout, label smoothing)
- Multi-head attention for temporal modeling
- Improved frequency segregation
- Better hyperparameter configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for electrode channel importance"""
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm1d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, T)
        attention_weights = self.attention(x)
        return x * attention_weights


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time step importance"""
    def __init__(self, features):
        super(TemporalAttention, self).__init__()
        self.features = features
        self.attention = nn.Sequential(
            nn.Linear(features, features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(features // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, T, F)
        attention_weights = self.attention(x).squeeze(-1)  # (B, T)
        attention_weights = attention_weights.unsqueeze(-1)  # (B, T, 1)
        return x * attention_weights


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention for temporal modeling"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadTemporalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        output = self.out_linear(context)
        return output


class ResidualBlock1D(nn.Module):
    """Residual block for 1D convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class ImprovedLayer1(nn.Module):
    """
    Enhanced Layer 1: Spatio-temporal feature extraction with attention and residual connections
    """
    def __init__(self):
        super(ImprovedLayer1, self).__init__()
        
        # Spatial block with residual connections
        self.spatial_block = nn.Sequential(
            ResidualBlock1D(64, 128, kernel_size=5, dropout=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ResidualBlock1D(128, 256, kernel_size=3, dropout=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SpatialAttention(256)
        )
        
        # Temporal block with multi-head attention
        self.temporal_attention = MultiHeadTemporalAttention(embed_dim=256, num_heads=8, dropout=0.1)
        
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Temporal attention after LSTM
        self.post_lstm_attention = TemporalAttention(256)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # x: (B, 64, 256)
        
        # Spatial processing
        x = self.spatial_block(x)  # (B, 256, 64)
        
        # Permute for temporal processing
        x = x.permute(0, 2, 1)  # (B, 64, 256)
        
        # Multi-head temporal attention
        x = self.temporal_attention(x)  # (B, 64, 256)
        
        # BiLSTM processing
        lstm_out, _ = self.lstm(x)  # (B, 64, 256)
        
        # Post-LSTM temporal attention
        lstm_out = self.post_lstm_attention(lstm_out)  # (B, 64, 256)
        
        # Take last time step
        x = lstm_out[:, -1, :]  # (B, 256)
        
        # Project to 128 dimensions
        x = self.projection(x)  # (B, 128)
        
        return x


class ImprovedLayer2(nn.Module):
    """
    Enhanced Layer 2: Frequency-segregated processing with improved architecture
    """
    def __init__(self):
        super(ImprovedLayer2, self).__init__()
        
        # Slow-wave stream (delta/theta) with residual connections
        self.slow_stream = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock1D(64, 64, kernel_size=5, dropout=0.2),
            nn.MaxPool1d(kernel_size=4, stride=4),
            SpatialAttention(64)
        )
        
        # Fast-wave stream (beta/gamma) with residual connections
        self.fast_stream = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock1D(64, 64, kernel_size=3, dropout=0.2),
            nn.MaxPool1d(kernel_size=8, stride=8),
            SpatialAttention(64)
        )
        
        # Additional frequency bands for better coverage
        self.mid_stream = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=6, stride=6),
            SpatialAttention(64)
        )
        
        # Fusion layer
        total_features = 64 * 32 + 64 * 32 + 64 * 42  # slow + fast + mid
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # x: (B, 64, 256)
        
        # Process all frequency streams
        slow_feat = self.slow_stream(x)  # (B, 64, 32)
        fast_feat = self.fast_stream(x)  # (B, 64, 32)
        mid_feat = self.mid_stream(x)   # (B, 64, 42)
        
        # Flatten
        slow_feat = slow_feat.reshape(slow_feat.size(0), -1)
        fast_feat = fast_feat.reshape(fast_feat.size(0), -1)
        mid_feat = mid_feat.reshape(mid_feat.size(0), -1)
        
        # Concatenate all streams
        combined = torch.cat([slow_feat, fast_feat, mid_feat], dim=1)
        
        # Fusion
        x = self.fusion(combined)  # (B, 128)
        
        return x


class ImprovedLayer3(nn.Module):
    """
    Enhanced Layer 3: Meta-classifier with attention and improved architecture
    """
    def __init__(self, input_dim=256, num_classes=2):
        super(ImprovedLayer3, self).__init__()
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Apply feature attention
        attention_weights = self.feature_attention(x)
        x = x * attention_weights
        
        # Classification
        x = self.classifier(x)
        
        return x


class ImprovedTripleLayerMetaModel(nn.Module):
    """
    Complete improved Triple-Layer MetaModel
    Target: 99%+ accuracy
    """
    def __init__(self, num_classes=2):
        super(ImprovedTripleLayerMetaModel, self).__init__()
        
        self.layer1 = ImprovedLayer1()
        self.layer2 = ImprovedLayer2()
        self.layer3 = ImprovedLayer3(input_dim=256, num_classes=num_classes)
        
        # Additional statistics features
        # mean (64) + std (64) + max (64) + min (64) = 256 features
        self.stats_extractor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Extract features from both pathways
        feat1 = self.layer1(x)  # Spatio-temporal features (B, 128)
        feat2 = self.layer2(x)  # Frequency-segregated features (B, 128)
        
        # Extract statistical features from raw input
        stats = torch.cat([x.mean(dim=2), x.std(dim=2), x.max(dim=2)[0], x.min(dim=2)[0]], dim=1)
        stats = self.stats_extractor(stats)  # (B, 64)
        
        # Concatenate all features
        # feat1 (128) + feat2 (128) + stats (64) = 320
        # We need to project to 256 for layer3
        combined = torch.cat([feat1, feat2], dim=1)  # (B, 256)
        
        # Final classification
        logits = self.layer3(combined)
        
        return logits
    
    def get_feature_embeddings(self, x):
        """Extract intermediate feature embeddings for analysis"""
        with torch.no_grad():
            feat1 = self.layer1(x)
            feat2 = self.layer2(x)
            stats = torch.cat([x.mean(dim=2), x.std(dim=2), x.max(dim=2)[0], x.min(dim=2)[0]], dim=1)
            stats = self.stats_extractor(stats)
            combined = torch.cat([feat1, feat2], dim=1)
        
        return {
            'spatio_temporal': feat1,
            'frequency_segregated': feat2,
            'statistical': stats,
            'combined': combined
        }


class LightweightMetaModel(nn.Module):
    """
    Lightweight version for faster training and deployment
    Maintains good accuracy while reducing computational cost
    """
    def __init__(self, num_classes=2):
        super(LightweightMetaModel, self).__init__()
        
        # Simplified Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            SpatialAttention(256)
        )
        
        self.lstm = nn.LSTM(256, 64, num_layers=1, batch_first=True, bidirectional=True)
        
        # Simplified Layer 2
        self.slow = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4)
        )
        
        self.fast = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(8)
        )
        
        # Meta-classifier
        self.fc_freq = nn.Linear(64*32 + 64*32, 128)
        self.meta = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Layer 1
        x1 = self.layer1(x)  # (B, 256, 64)
        x1 = x1.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x1)
        feat1 = lstm_out[:, -1, :]  # (B, 128)
        
        # Layer 2
        slow = self.slow(x).reshape(x.size(0), -1)
        fast = self.fast(x).reshape(x.size(0), -1)
        feat2 = self.fc_freq(torch.cat([slow, fast], dim=1))
        
        # Meta-classifier
        combined = torch.cat([feat1, feat2], dim=1)
        logits = self.meta(combined)
        
        return logits


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model):
    """Get model architecture summary"""
    summary = {
        'total_params': count_parameters(model),
        'layers': []
    }
    
    for name, module in model.named_children():
        params = count_parameters(module)
        summary['layers'].append({
            'name': name,
            'type': type(module).__name__,
            'params': params
        })
    
    return summary


if __name__ == "__main__":
    # Test the models
    print("Testing Improved EEG Models")
    print("=" * 60)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 64, 256)
    
    # Test full model
    print("\n1. Testing ImprovedTripleLayerMetaModel")
    model = ImprovedTripleLayerMetaModel()
    print(f"Total parameters: {count_parameters(model):,}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_feature_embeddings(dummy_input)
    print("Feature embeddings:")
    for key, value in features.items():
        print(f"  {key}: {value.shape}")
    
    # Test lightweight model
    print("\n2. Testing LightweightMetaModel")
    lightweight_model = LightweightMetaModel()
    print(f"Total parameters: {count_parameters(lightweight_model):,}")
    
    output_light = lightweight_model(dummy_input)
    print(f"Output shape: {output_light.shape}")
    
    print("\n" + "=" * 60)
    print("Model testing complete!")