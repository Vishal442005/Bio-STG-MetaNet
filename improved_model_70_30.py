"""
Improved Model with 70:30 Train/Test Split
Better architecture to prevent overfitting
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import gzip
import pandas as pd
from scipy import signal
import copy

print("="*60)
print("IMPROVED MODEL WITH 70:30 TRAIN/TEST SPLIT")
print("="*60)

# Configuration
NUM_FILES = 600  # Use all available data (300 per class)
TRAIN_TEST_SPLIT = 0.7  # 70% train, 30% test
EPOCHS = 50  # More epochs for better convergence
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # Lower learning rate for stability
DROPOUT_RATE = 0.5  # Higher dropout to prevent overfitting

print(f"\nConfiguration:")
print(f"  Total files: {NUM_FILES}")
print(f"  Train/Test split: {TRAIN_TEST_SPLIT*100}%/{(1-TRAIN_TEST_SPLIT)*100}%")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Dropout rate: {DROPOUT_RATE}")

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

# 70:30 Train/Test Split
print(f"\nSplitting data: {TRAIN_TEST_SPLIT*100}% train, {(1-TRAIN_TEST_SPLIT)*100}% test")
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=1-TRAIN_TEST_SPLIT, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}, Class distribution: Alcoholic={np.sum(y_train==1)}, Control={np.sum(y_train==0)}")
print(f"Test set: {X_test.shape}, Class distribution: Alcoholic={np.sum(y_test==1)}, Control={np.sum(y_test==0)}")

# Improved Model Architecture
class ImprovedEEGModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # Block 1: Spatial features with residual connections
        self.spatial_block = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Block 2: Temporal features
        self.temporal_block = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Deep features
        self.deep_block = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate final size after convolutions
        # Input: 64x256
        # After spatial: 128x256
        # After temporal: 256x64 (256 -> 128 -> 64)
        # After deep: 512x32 (64 -> 32)
        self.fc = nn.Sequential(
            nn.Linear(512 * 32, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # Spatial features
        x1 = self.spatial_block(x)
        
        # Residual connection
        x = x + x1
        
        # Temporal features
        x = self.temporal_block(x)
        
        # Deep features
        x = self.deep_block(x)
        
        # Flatten and classify
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Training
print("\nStarting training...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ImprovedEEGModel(dropout_rate=DROPOUT_RATE).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

best_test_acc = 0
best_model_state = None
train_acc_history = []
test_acc_history = []
train_loss_history = []
test_loss_history = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
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
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
    
    train_acc = 100 * train_correct / train_total
    train_loss_avg = train_loss / len(train_loader)
    
    # Testing
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = 100 * test_correct / test_total
    test_loss_avg = test_loss / len(test_loader)
    
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    train_loss_history.append(train_loss_avg)
    test_loss_history.append(test_loss_avg)
    
    scheduler.step(test_acc)
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_state = copy.deepcopy(model.state_dict())
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{EPOCHS}: Train Acc = {train_acc:6.2f}%, Test Acc = {test_acc:6.2f}%, "
              f"Train Loss = {train_loss_avg:.4f}, Test Loss = {test_loss_avg:.4f}")

# Load best model
model.load_state_dict(best_model_state)

# Final evaluation
model.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_test_acc = 100 * test_correct / test_total

# Measure inference time
inference_times = []
model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        start = time.time()
        _ = model(inputs)
        end = time.time()
        inference_times.append((end - start) * 1000)

avg_inference_time = np.mean(inference_times)

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Best Test Accuracy: {best_test_acc:.2f}%")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
print(f"Training Accuracy (final): {train_acc:.2f}%")
print(f"Mean Inference Time: {avg_inference_time:.2f} ± {np.std(inference_times):.2f} ms")

# Save outputs
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_acc_history, label='Training Accuracy', marker='o')
plt.plot(test_acc_history, label='Test Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training vs Test Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_loss_history, label='Training Loss', marker='o')
plt.plot(test_loss_history, label='Test Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Test Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Control', 'Alcoholic'])
plt.yticks([0, 1], ['Control', 'Alcoholic'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig('plots/improved_model_results.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to plots/improved_model_results.png")

# Save results table
import pandas as pd
results_df = pd.DataFrame([{
    'Train_Accuracy': train_acc,
    'Test_Accuracy': final_test_acc,
    'Best_Test_Accuracy': best_test_acc,
    'Inference_Time_ms': avg_inference_time,
    'Train_Samples': len(X_train),
    'Test_Samples': len(X_test)
}])
results_df.to_csv('results/improved_model_results.csv', index=False)
print(f"Results table saved to results/improved_model_results.csv")

# Save best model
torch.save(best_model_state, 'models/improved_model.pth')
print(f"Best model saved to models/improved_model.pth")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")