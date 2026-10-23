"""
Simple training script to test with reasonable data size
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import glob
import gzip
import pandas as pd
from scipy import signal

print("Simple EEG Training Test")
print("="*60)

# Load a small sample of data for testing
print("\nLoading data (100 files total)...")

X_list = []
y_list = []

# Get files by class
train_files_0 = glob.glob('SMNI_CMI_TRAIN/co2c*/**/*.gz', recursive=True)
train_files_1 = glob.glob('SMNI_CMI_TRAIN/co2a*/**/*.gz', recursive=True)

print(f"Control files: {len(train_files_0)}")
print(f"Alcoholic files: {len(train_files_1)}")

# Take 50 from each class
import random
random.seed(42)
sampled_0 = random.sample(train_files_0, min(50, len(train_files_0)))
sampled_1 = random.sample(train_files_1, min(50, len(train_files_1)))

print(f"Loading {len(sampled_0)} control files...")
for i, gz_file in enumerate(sampled_0):
    try:
        with gzip.open(gz_file, 'rt') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line.startswith('S1') or line.startswith('FP1'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    trial_num = int(parts[0])
                    sensor_pos = parts[1]
                    sample_idx = int(parts[2])
                    voltage = float(parts[3])
                    data.append([trial_num, sensor_pos, sample_idx, voltage])
                except:
                    continue
        
        df = pd.DataFrame(data, columns=['trial', 'sensor', 'sample', 'voltage'])
        if len(df) == 0:
            print(f"  No valid data in {gz_file}")
            continue
        pivot_df = df.pivot(index='sensor', columns='sample', values='voltage')
        
        # Ensure 64x256
        if pivot_df.shape[0] != 64 or pivot_df.shape[1] != 256:
            if pivot_df.shape[0] < 64:
                missing = 64 - pivot_df.shape[0]
                zeros = pd.DataFrame(0, index=range(missing), columns=pivot_df.columns)
                pivot_df = pd.concat([pivot_df, zeros], ignore_index=True)
            if pivot_df.shape[1] < 256:
                missing = 256 - pivot_df.shape[1]
                zeros = pd.DataFrame(0, index=pivot_df.index, columns=range(missing))
                pivot_df = pd.concat([pivot_df, zeros], axis=1)
        
        X_list.append(pivot_df.values)
        y_list.append(0)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sampled_0)} control files")
    except Exception as e:
        print(f"Error: {e}")

print(f"Loading {len(sampled_1)} alcoholic files...")
for i, gz_file in enumerate(sampled_1):
    try:
        with gzip.open(gz_file, 'rt') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line.startswith('S1') or line.startswith('FP1'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    trial_num = int(parts[0])
                    sensor_pos = parts[1]
                    sample_idx = int(parts[2])
                    voltage = float(parts[3])
                    data.append([trial_num, sensor_pos, sample_idx, voltage])
                except:
                    continue
        
        df = pd.DataFrame(data, columns=['trial', 'sensor', 'sample', 'voltage'])
        if len(df) == 0:
            print(f"  No valid data in {gz_file}")
            continue
        pivot_df = df.pivot(index='sensor', columns='sample', values='voltage')
        
        # Ensure 64x256
        if pivot_df.shape[0] != 64 or pivot_df.shape[1] != 256:
            if pivot_df.shape[0] < 64:
                missing = 64 - pivot_df.shape[0]
                zeros = pd.DataFrame(0, index=range(missing), columns=pivot_df.columns)
                pivot_df = pd.concat([pivot_df, zeros], ignore_index=True)
            if pivot_df.shape[1] < 256:
                missing = 256 - pivot_df.shape[1]
                zeros = pd.DataFrame(0, index=pivot_df.index, columns=range(missing))
                pivot_df = pd.concat([pivot_df, zeros], axis=1)
        
        X_list.append(pivot_df.values)
        y_list.append(1)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sampled_1)} alcoholic files")
    except Exception as e:
        print(f"Error: {e}")

X = np.array(X_list)
y = np.array(y_list)

print(f"\nLoaded data shape: {X.shape}")
print(f"Class distribution: Alcoholic={np.sum(y==1)}, Control={np.sum(y==0)}")

# Simple preprocessing
print("\nPreprocessing...")
X_filtered = np.zeros_like(X)
nyquist = 256 / 2
low = 0.5 / nyquist
high = 100 / nyquist
b, a = signal.butter(4, [low, high], btype='band')

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X_filtered[i, j, :] = signal.filtfilt(b, a, X[i, j, :])

# Normalize
X_normalized = np.zeros_like(X_filtered)
for i in range(X_filtered.shape[0]):
    for j in range(X_filtered.shape[1]):
        mean = np.mean(X_filtered[i, j, :])
        std = np.std(X_filtered[i, j, :])
        if std > 1e-6:
            X_normalized[i, j, :] = (X_filtered[i, j, :] - mean) / std
        else:
            X_normalized[i, j, :] = X_filtered[i, j, :] - mean

print("Preprocessing complete")

# Simple model for testing
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Sequential(
            nn.Linear(256 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# Training
print("\nStarting training...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = TensorDataset(torch.FloatTensor(X_normalized), torch.LongTensor(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_val_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_normalized, y)):
    print(f"\nFold {fold + 1}/5")
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    for epoch in range(10):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Val Acc = {val_acc:.2f}%")
    
    all_val_accs.append(best_val_acc)
    print(f"  Best Val Acc: {best_val_acc:.2f}%")

print(f"\nFinal Results:")
print(f"Mean Validation Accuracy: {np.mean(all_val_accs):.2f}% ± {np.std(all_val_accs):.2f}%")

# Save plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), all_val_accs, color='steelblue', alpha=0.7)
plt.axhline(y=np.mean(all_val_accs), color='red', linestyle='--', label=f'Mean: {np.mean(all_val_accs):.2f}%')
plt.xlabel('Fold')
plt.ylabel('Validation Accuracy (%)')
plt.title('Per-Fold Validation Accuracy')
plt.legend()
plt.grid(True, axis='y')

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/simple_training_results.png')
print(f"\nPlot saved to plots/simple_training_results.png")