import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, Subset

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

import os



# =====================================================================

# LAYER 1: BASELINE DEEP LEARNER (CNN-LSTM)

# Extracts general spatio-temporal features across all EEG bands.

# =====================================================================

class Baseline_CNNLSTM(nn.Module):

    def __init__(self):

        super(Baseline_CNNLSTM, self).__init__()

        # Spatial Block

        self.conv_block = nn.Sequential(

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2),

            

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Dropout(0.2)

        )

        # Temporal Block

        self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=2, 

                            batch_first=True, bidirectional=True, dropout=0.2)



    def forward(self, x):

        x = self.conv_block(x)

        x = x.permute(0, 2, 1) # (Batch, Seq, Features)

        lstm_out, _ = self.lstm(x)

        return lstm_out[:, -1, :] # Last time step (Batch, 128)



# =====================================================================

# LAYER 2: MARGINAL / SEGREGATED LEARNER

# Focuses on explicit wave band segregation using parallel large/small kernels.

# Demonstrates the "Marginal Quantization" concept.

# =====================================================================

class Marginal_Segregated_CNN(nn.Module):

    def __init__(self):

        super(Marginal_Segregated_CNN, self).__init__()

        # Delta/Theta focus (Large kernel to capture slow, long waves)

        self.slow = nn.Sequential(

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, stride=2, padding=7),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4) # Heavily quantize

        )

        # Beta/Gamma focus (Small kernel to capture fast, rapid waves)

        self.fast = nn.Sequential(

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(kernel_size=8, stride=8) # Quantize to match spatial dims

        )

        # Flatten and combine

        # Input has 256 time steps. 

        # slow: 256 / 2 (stride) = 128. 128 / 4 (pool) = 32.

        # fast: 256 / 1 (stride) = 256. 256 / 8 (pool) = 32.

        self.fc = nn.Sequential(

            nn.Linear(64*32 + 64*32, 128), 

            nn.ReLU(),

            nn.Dropout(0.2)

        )



    def forward(self, x):

        slow_feat = self.slow(x)

        fast_feat = self.fast(x)

        

        slow_feat = slow_feat.reshape(slow_feat.size(0), -1)

        fast_feat = fast_feat.reshape(fast_feat.size(0), -1)

        

        combined = torch.cat((slow_feat, fast_feat), dim=1)

        return self.fc(combined) # Output: (Batch, 128)



# =====================================================================

# LAYER 3: META-CLASSIFIER

# Evaluates outputs from Layer 1 and Layer 2 to make final decision.

# Unfiltered margin ML combined with L1 & L2 for prediction.

# =====================================================================

class TripleLayer_MetaModel(nn.Module):

    def __init__(self, num_classes=2):

        super(TripleLayer_MetaModel, self).__init__()

        self.layer1 = Baseline_CNNLSTM()

        self.layer2 = Marginal_Segregated_CNN()

        

        # Meta-Classifier takes concatenated features (128 + 128 = 256)

        self.meta = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, num_classes)

        )



    def forward(self, x):

        feat1 = self.layer1(x)  # General spatio-temporal features

        feat2 = self.layer2(x)  # Segregated frequency features

        

        # Meta-learning concatenation

        meta_features = torch.cat((feat1, feat2), dim=1)

        

        # Final Decision

        logits = self.meta(meta_features)

        return logits



# =====================================================================

# 10-FOLD CROSS VALIDATION & TRAINING PIPELINE

# =====================================================================

if __name__ == "__main__":

    print("Initializing Triple-Layer Meta-Learning Framework...")

    

    # Attempt to load data

    try:

        X = np.load('X_filtered.npy')

        y = np.load('y_train.npy')

        print(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")

    except FileNotFoundError:

        print("Data files not found. Creating dummy data for demonstration...")

        X = np.random.randn(600, 64, 256).astype(np.float32)

        y = np.random.randint(0, 2, size=(600,)).astype(np.int64)



    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))

    

    # Training Parameters

    epochs = 30

    batch_size = 64

    k_folds = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    

    # Tracking metrics across all folds

    avg_train_acc_per_epoch = np.zeros(epochs)

    avg_val_acc_per_epoch = np.zeros(epochs)

    

    overall_best_acc = 0.0



    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        print(f"\n--- Fold {fold + 1}/{k_folds} ---")

        

        train_sub = Subset(dataset, train_idx)

        val_sub = Subset(dataset, val_idx)

        

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        

        # Initialize model for this fold

        model = TripleLayer_MetaModel(num_classes=2).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=0.000125, weight_decay=5e-4)

        last_accuracy = overall_best_acc

        for epoch in range(epochs):

            # Training Phase

            model.train()

            correct_train = 0

            total_train = 0

            

            for inputs, labels in train_loader:

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                

                _, predicted = torch.max(outputs.data, 1)

                total_train += labels.size(0)

                correct_train += (predicted == labels).sum().item()

                

            train_acc = 100 * correct_train / total_train

            avg_train_acc_per_epoch[epoch] += train_acc

            

            # Validation Phase

            model.eval()

            correct_val = 0

            total_val = 0

            

            with torch.no_grad():

                for inputs, labels in val_loader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)

                    _, predicted = torch.max(outputs.data, 1)

                    total_val += labels.size(0)

                    correct_val += (predicted == labels).sum().item()

                    

            val_acc = 100 * correct_val / total_val

            avg_val_acc_per_epoch[epoch] += val_acc

            

            # Save the overall best model across ALL folds

            if val_acc >= overall_best_acc:

                overall_best_acc = val_acc

                torch.save(model.state_dict(), 'best_modal.pth')

                print(f"    [Epoch {epoch+1}] New Best Validation Accuracy: {val_acc:.2f}% -> Saved best_modal.pth")
                print(f"    [Epoch {epoch+1}] Training Accuracy: {train_acc:.2f}% | Validation Accuracy: {val_acc:.2f}%")
        if last_accuracy > overall_best_acc:
            fold = fold - 1
            print("heldback")

                

    # Calculate Averages

    avg_train_acc_per_epoch /= k_folds

    avg_val_acc_per_epoch /= k_folds

    

    print(f"\n10-Fold Cross Validation Complete. Overall Best Accuracy: {overall_best_acc:.2f}%")

    

    # Plotting Train vs Validation Accuracy

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, epochs + 1), avg_train_acc_per_epoch, label='Average Training Accuracy', marker='o')

    plt.plot(range(1, epochs + 1), avg_val_acc_per_epoch, label='Average Validation Accuracy', marker='s')

    

    plt.title('10-Fold CV: Training vs Validation Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy (%)')

    plt.legend()

    plt.grid(True)

    plt.savefig('Train_Val_Accuracy_Plot.png')

    plt.show()


    print("Plot saved as 'Train_Val_Accuracy_Plot.png'.")
