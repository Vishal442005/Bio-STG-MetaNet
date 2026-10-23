"""
Final Working Training Script
Based on the working simple_train.py that achieved 77% accuracy
This will use proper data loading and generate organized outputs
"""

import os
import time
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
import copy

print("="*60)
print("FINAL WORKING EEG TRAINING")
print("="*60)

# Configuration
NUM_FILES = 600  # 300 per class - use all available data
N_FOLDS = 5  # Fewer folds for larger validation sets
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 2e-4

print(f"\nConfiguration:")
print(f"  Files per class: {NUM_FILES//2}")
print(f"  Total files: {NUM_FILES}")
print(f"  Folds: {N_FOLDS}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")

# Load data
print("\nLoading data...")
X_list = []
y_list = []

# Get files by class
train_files_0 = glob.glob('SMNI_CMI_TRAIN/co2c*/**/*.gz', recursive=True)
train_files_1 = glob.glob('SMNI_CMI_TRAIN/co2a*/**/*.gz', recursive=True)

print(f"Control files available: {len(train_files_0)}")
print(f"Alcoholic files available: {len(train_files_1)}")

# Sample files
import random
random.seed(42)
files_per_class = min(NUM_FILES//2, len(train_files_0), len(train_files_1))
sampled_0 = random.sample(train_files_0, files_per_class)
sampled_1 = random.sample(train_files_1, files_per_class)

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
        
        if (i + 1) % 50 == 0:
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
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sampled_1)} alcoholic files")
    except Exception as e:
        print(f"Error: {e}")

X = np.array(X_list)
y = np.array(y_list)

print(f"\nLoaded data shape: {X.shape}")
print(f"Class distribution: Alcoholic={np.sum(y==1)}, Control={np.sum(y==0)}")

# Preprocessing
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

# Model
class WorkingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# Training
print("\nStarting 10-fold cross-validation...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = TensorDataset(torch.FloatTensor(X_normalized), torch.LongTensor(y))
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

all_fold_results = []
best_overall_acc = 0
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_normalized, y)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = WorkingCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_acc = 0
    patience_counter = 0
    train_acc_history = []
    val_acc_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        train_acc_history.append(train_acc)
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        val_acc_history.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:2d}/{EPOCHS}: Train Acc = {train_acc:6.2f}%, Val Acc = {val_acc:6.2f}%")
        
        if patience_counter >= 5:
            break
    
    model.load_state_dict(best_state)
    
    # Measure inference time
    model.eval()
    inference_times = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            start = time.time()
            _ = model(inputs)
            end = time.time()
            inference_times.append((end - start) * 1000)
    
    avg_inference_time = np.mean(inference_times)
    
    all_fold_results.append({
        'fold': fold + 1,
        'best_val_acc': best_val_acc,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        'inference_time_ms': avg_inference_time
    })
    
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    
    if best_val_acc > best_overall_acc:
        best_overall_acc = best_val_acc
        best_model_state = copy.deepcopy(model.state_dict)

# Calculate statistics
accuracies = [r['best_val_acc'] for r in all_fold_results]
inference_times = [r['inference_time_ms'] for r in all_fold_results]

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Mean Validation Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
print(f"Best Overall Accuracy: {best_overall_acc:.2f}%")
print(f"Mean Inference Time: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms")

# Save outputs
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for fold_result in all_fold_results:
    plt.plot(fold_result['train_acc_history'], alpha=0.6, label=f'Fold {fold_result["fold"]}')
plt.plot(fold_result['val_acc_history'], alpha=0.6, linestyle='--', label=f'Fold {fold_result["fold"]} (val)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Progress Across Folds')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.subplot(1, 2, 2)
fold_nums = [r['fold'] for r in all_fold_results]
fold_accs = [r['best_val_acc'] for r in all_fold_results]
plt.bar(fold_nums, fold_accs, color='steelblue', alpha=0.7)
plt.axhline(y=np.mean(fold_accs), color='red', linestyle='--', label=f'Mean: {np.mean(fold_accs):.2f}%')
plt.xlabel('Fold')
plt.ylabel('Best Validation Accuracy (%)')
plt.title('Per-Fold Best Validation Accuracy')
plt.legend()
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('plots/final_training_results.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to plots/final_training_results.png")

# Save results table
import pandas as pd
results_df = pd.DataFrame([{
    'Fold': r['fold'],
    'Best_Val_Accuracy': r['best_val_acc'],
    'Inference_Time_ms': r['inference_time_ms']
} for r in all_fold_results])
results_df.to_csv('results/final_results.csv', index=False)
print(f"Results table saved to results/final_results.csv")

# Save best model
torch.save(best_model_state, 'models/best_working_model.pth')
print(f"Best model saved to models/best_working_model.pth")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")