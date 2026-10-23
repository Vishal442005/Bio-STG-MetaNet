import os
import gzip
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==========================================
class EEGNet_Hybrid(nn.Module):
    def __init__(self, num_classes=2):
        super(EEGNet_Hybrid, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Dropout(0.4)
        )
        
        self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute(0, 2, 1) 
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] 
        logits = self.fc_block(last_out)
        return logits

class Baseline_CNNLSTM(nn.Module):
    def __init__(self):
        super(Baseline_CNNLSTM, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.2)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]

class Marginal_Segregated_CNN(nn.Module):
    def __init__(self):
        super(Marginal_Segregated_CNN, self).__init__()
        self.slow = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        self.fast = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*32 + 64*32, 128), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        slow_feat = self.slow(x)
        fast_feat = self.fast(x)
        slow_feat = slow_feat.reshape(slow_feat.size(0), -1)
        fast_feat = fast_feat.reshape(fast_feat.size(0), -1)
        combined = torch.cat((slow_feat, fast_feat), dim=1)
        return self.fc(combined)

class TripleLayer_MetaModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TripleLayer_MetaModel, self).__init__()
        self.layer1 = Baseline_CNNLSTM()
        self.layer2 = Marginal_Segregated_CNN()
        self.meta = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat1 = self.layer1(x) 
        feat2 = self.layer2(x) 
        meta_features = torch.cat((feat1, feat2), dim=1)
        logits = self.meta(meta_features)
        return logits


# ==========================================
# 2. DATA LOADING & DSP FUNCTIONS
# ==========================================
def read_eeg_file(filepath):
    """Parses raw EEG .gz files."""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f if not line.startswith('#') and line.strip()]
        
    columns = ['trial_number', 'sensor_position', 'sample_num', 'sensor_value']
    df = pd.DataFrame(lines, columns=columns)
    
    df['sensor_value'] = df['sensor_value'].astype(float)
    df['sample_num'] = df['sample_num'].astype(int)
    return df

def apply_bulk_bandpass(data, fs=256, lowcut=0.5, highcut=100, order=4):
    """Applies a zero-phase Butterworth filter across the time axis."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

# ==========================================
# 3. VISUALIZATION MODULES
# ==========================================
def plot_side_by_side(raw_signal, filtered_signal, channel_idx=0, iteration_num=1):
    """Plots the raw vs filtered signal side-by-side. Blocks execution until closed."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.canvas.manager.set_window_title(f'Iteration {iteration_num} - Signal Processing')
    
    axes[0].plot(raw_signal, color='red', linewidth=1.2)
    axes[0].set_title(f'Raw EEG Signal (Electrode {channel_idx})')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Voltage (\u03BCV)')
    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(filtered_signal, color='blue', linewidth=1.2)
    axes[1].set_title(f'Filtered EEG Signal (0.5-100Hz)')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Voltage (\u03BCV)')
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show() # Halts script here until user closes window

def plot_running_accuracy(history, is_correct):
    """Plots the live accuracy trend. Blocks execution until closed."""
    plt.figure(figsize=(10, 5))
    plt.gcf().canvas.manager.set_window_title('Running Accuracy Tracker')
    
    # Plot the line
    x_axis = range(1, len(history) + 1)
    plt.plot(x_axis, history, marker='o', linestyle='-', color='royalblue', linewidth=2)
    
    # Highlight the most recent point (Green if correct, Red if wrong)
    point_color = 'limegreen' if is_correct else 'crimson'
    plt.scatter(len(history), history[-1], color=point_color, s=150, zorder=5, 
                edgecolor='black', label='Latest Prediction')
    
    plt.title(f'Live Model Accuracy: {history[-1]:.2f}%')
    plt.xlabel('Test Iteration')
    plt.ylabel('Accuracy (%)')
    plt.ylim(-5, 105) # Fixed Y-axis from 0 to 100
    plt.axhline(100, color='black', linewidth=0.1,linestyle='--', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show() # Halts script here until user closes window

# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    # --- CONFIGURATION ---
    iteration = int(input("enter no. of iterations : "))  # Number of times the loop will run
    test_dir = r"C:\Users\Visha\Desktop\Internshhip\SMNI_CMI_TEST"
    # ---------------------

    # Setup Device & Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    try:
        c = int(input("if you want phase 1 model enter 1 , if phase 2 modal enter 2 and if phase 0 modal enter 0: "))
        k = 0
        
        # We must initialize the CORRECT architecture before loading weights into it
        if c == 0:
            model = EEGNet_Hybrid(num_classes=2).to(device)
            model.load_state_dict(torch.load('bestnotemodel.pt', map_location=device))
            k = 0
        elif c == 1:
            # THIS IS THE NEW MODEL
            model = TripleLayer_MetaModel(num_classes=2).to(device)
            model.load_state_dict(torch.load('best_modal.pth', map_location=device))
            k = 1
        elif c == 2:
            model = EEGNet_Hybrid(num_classes=2).to(device)
            model.load_state_dict(torch.load('best_eeg_model.pth', map_location=device))
            k = 2
        else:
            print("what the hell did you typed are you blind")
            return
            
        model.eval()
        print(f"Successfully loaded mode: {k}")
        
    except FileNotFoundError:
        print("Error: model weights file not found. Ensure .pth or .pt file is in the same directory.")
        return

    if not os.path.exists(test_dir):
        print(f"Error: Folder '{test_dir}' not found.")
        return

    # Collect valid files
    print("Indexing dataset...")
    valid_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.gz') and not file.endswith('.tar.gz'):
                valid_files.append(os.path.join(root, file))

    if not valid_files:
        print("No valid EEG files found.")
        return

    # Trackers
    correct_count = 0
    accuracy_history = []
    
    # --- INTERACTIVE TESTING LOOP ---
    for i in range(1, iteration + 1):
        print(f"\n{'='*40}")
        print(f" ITERATION {i} OF {iteration}")
        print(f"{'='*40}")
        
        # 1. Select Random File
        random_filepath = random.choice(valid_files)
        folder_name = os.path.basename(os.path.dirname(random_filepath))
        actual_label = "Alcoholic Subject" if 'co2a' in folder_name else "Control Subject"
        print(f"Processing: {os.path.basename(random_filepath)}")
        
        # 2. Load & Filter Data
        df = read_eeg_file(random_filepath)
        raw_matrix = df.pivot(index='sensor_position', columns='sample_num', values='sensor_value').values
        
        if raw_matrix.shape != (64, 256):
            print(f"Skipping corrupt file (Shape {raw_matrix.shape}).")
            continue
            
        filtered_matrix = apply_bulk_bandpass(raw_matrix)
        
        # 3. Model Inference (Keeping the .copy() fix!)
        input_tensor = torch.FloatTensor(filtered_matrix.copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        prediction_label = "Alcoholic Subject" if predicted_class == 1 else "Control Subject"
        is_correct = (actual_label == prediction_label)
        
        # Update metrics
        if is_correct: correct_count += 1
        current_accuracy = (correct_count / i) * 100
        accuracy_history.append(current_accuracy)
        
        if current_accuracy < 0.97:
            # Terminal Output
            print(">> Close the Signal Plot to view the Accuracy Tracker...")
            
        print(f"True Label: {actual_label}")
        print(f"Prediction: {prediction_label}")
        print(f"Current Accuracy: {current_accuracy:.2f}%")
        
        # 4. Render UI Plots (Sequential Blocking)
        if i >= iteration:
            plot_side_by_side(raw_matrix[0, :], filtered_matrix[0, :], channel_idx=0, iteration_num=i)
            print(">> Close the Accuracy Tracker to start the next iteration...")
            plot_running_accuracy(accuracy_history, is_correct)
        
    print("\nTest sequence complete!")

if __name__ == "__main__":
    main()