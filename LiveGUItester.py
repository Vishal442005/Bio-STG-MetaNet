import os
import gzip
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.signal import butter, filtfilt

# ==========================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==========================================
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
            nn.Dropout(0.2)
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
            nn.Dropout(0.2)
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
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat1 = self.layer1(x) 
        feat2 = self.layer2(x) 
        meta_features = torch.cat((feat1, feat2), dim=1)
        return self.meta(meta_features)

# ==========================================
# 2. DATA LOADING & DSP FUNCTIONS
# ==========================================
def read_eeg_file(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f if not line.startswith('#') and line.strip()]
        
    columns = ['trial_number', 'sensor_position', 'sample_num', 'sensor_value']
    df = pd.DataFrame(lines, columns=columns)
    df['sensor_value'] = df['sensor_value'].astype(float)
    df['sample_num'] = df['sample_num'].astype(int)
    return df

def apply_bulk_bandpass(data, fs=256, lowcut=0.5, highcut=100, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

# ==========================================
# 3. ADVANCED GUI IMPLEMENTATION
# ==========================================
class SetupWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeuroDiagnose - Clinical Session Setup")
        self.geometry("450x450")
        self.configure(bg="#2c3e50")
        self.resizable(False, False)
        
        # Modern styling
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        style.configure("TLabel", background="#2c3e50", foreground="#ecf0f1", font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#3498db")
        style.configure("TCheckbutton", background="#2c3e50", foreground="#ecf0f1", font=("Segoe UI", 10))
        style.configure("Action.TButton", font=("Segoe UI", 12, "bold"), background="#27ae60", foreground="white")
        style.map("Action.TButton", background=[("active", "#2ecc71")])

        self.user_data = None
        
        main_frame = tk.Frame(self, bg="#2c3e50", padx=30, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="System Initialization", style="Header.TLabel").pack(pady=(0, 20))
        
        # Fields
        ttk.Label(main_frame, text="Clinician / User Name:").pack(anchor="w")
        self.name_entry = ttk.Entry(main_frame, width=40, font=("Segoe UI", 11))
        self.name_entry.pack(pady=(0, 15))
        
        ttk.Label(main_frame, text="Session ID / Patient Ref:").pack(anchor="w")
        self.session_entry = ttk.Entry(main_frame, width=40, font=("Segoe UI", 11))
        self.session_entry.pack(pady=(0, 15))

        ttk.Label(main_frame, text="Clinical Notes:").pack(anchor="w")
        self.notes_entry = ttk.Entry(main_frame, width=40, font=("Segoe UI", 11))
        self.notes_entry.pack(pady=(0, 20))
        
        self.default_var = tk.BooleanVar()
        self.default_cb = ttk.Checkbutton(main_frame, text="Use Default Demo Profile", variable=self.default_var, command=self.toggle_default)
        self.default_cb.pack(anchor="w", pady=(0, 25))
        
        self.submit_btn = ttk.Button(main_frame, text="Launch Dashboard \u279C", style="Action.TButton", command=self.submit)
        self.submit_btn.pack(fill=tk.X, ipady=5)

    def toggle_default(self):
        if self.default_var.get():
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, "Dr. Default")
            self.session_entry.delete(0, tk.END)
            self.session_entry.insert(0, f"SES-{datetime.datetime.now().strftime('%Y%m%d')}")
            self.notes_entry.delete(0, tk.END)
            self.notes_entry.insert(0, "Demo automated testing.")
            for e in [self.name_entry, self.session_entry, self.notes_entry]:
                e.config(state='disabled')
        else:
            for e in [self.name_entry, self.session_entry, self.notes_entry]:
                e.config(state='normal')
                e.delete(0, tk.END)

    def submit(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter a user name or use default profile.")
            return
        
        self.user_data = {
            "name": name,
            "session": self.session_entry.get().strip(),
            "notes": self.notes_entry.get().strip(),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.destroy()

class EEGTestDashboard(tk.Tk):
    def __init__(self, user_data, valid_files, model, device):
        super().__init__()
        self.title("Triple-Layer MetaModel | Clinical Dashboard")
        self.geometry("1400x900")
        self.configure(bg="#ecf0f1")
        
        self.user_data = user_data
        self.valid_files = valid_files
        self.model = model
        self.device = device
        
        # Statistics
        self.total_tests = 0
        self.correct_preds = 0
        
        # GUI state
        self.current_signals = []
        self.figures = []
        
        # Theming matplotlib
        plt.style.use('ggplot')
        
        self.setup_styles()
        self.build_ui()
        self.load_random_signals()
        
    def setup_styles(self):
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        
        style.configure("Sidebar.TFrame", background="#2c3e50")
        style.configure("Sidebar.TLabel", background="#2c3e50", foreground="white", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#2c3e50", foreground="#3498db", font=("Segoe UI", 16, "bold"))
        style.configure("Card.TFrame", background="white", relief="raised", borderwidth=1)
        style.configure("CardTitle.TLabel", background="white", foreground="#2c3e50", font=("Segoe UI", 12, "bold"))
        style.configure("CardValue.TLabel", background="white", foreground="#e74c3c", font=("Segoe UI", 24, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), background="#3498db", foreground="white")
        style.map("Primary.TButton", background=[("active", "#2980b9")])

    def build_ui(self):
        # 1. Sidebar (Left)
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="NeuroDiagnose", style="Title.TLabel").pack(pady=(20, 5))
        ttk.Label(sidebar, text=f"User: {self.user_data['name']}", style="Sidebar.TLabel").pack(anchor="w", padx=15, pady=2)
        ttk.Label(sidebar, text=f"Session: {self.user_data['session']}", style="Sidebar.TLabel").pack(anchor="w", padx=15, pady=2)
        
        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, padx=15, pady=15)
        
        # Controls
        ttk.Label(sidebar, text="Electrode to View:", style="Sidebar.TLabel").pack(anchor="w", padx=15, pady=5)
        self.channel_var = tk.StringVar(value="Channel 0 (Fp1)")
        channels = [f"Channel {i}" for i in range(64)]
        channels[0] = "Channel 0 (Fp1)"
        channels[28] = "Channel 28 (Cz)"
        channels[61] = "Channel 61 (O1)"
        
        self.channel_cb = ttk.Combobox(sidebar, textvariable=self.channel_var, values=channels, state="readonly", font=("Segoe UI", 10))
        self.channel_cb.pack(fill=tk.X, padx=15, pady=(0, 20))
        self.channel_cb.bind("<<ComboboxSelected>>", lambda e: self.replot_signals())

        ttk.Button(sidebar, text="\u21bb Fetch New Signals", style="Primary.TButton", command=self.load_random_signals).pack(fill=tk.X, padx=15, pady=10, ipady=5)
        
        # Stats Display in Sidebar
        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, padx=15, pady=15)
        ttk.Label(sidebar, text="Session Stats", style="Sidebar.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15)
        
        self.lbl_total = ttk.Label(sidebar, text="Tested: 0", style="Sidebar.TLabel")
        self.lbl_total.pack(anchor="w", padx=15, pady=5)
        self.lbl_acc = ttk.Label(sidebar, text="Accuracy: 0.0%", style="Sidebar.TLabel")
        self.lbl_acc.pack(anchor="w", padx=15, pady=5)

        # 2. Main Content Area (Right)
        main_area = tk.Frame(self, bg="#ecf0f1")
        main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Top Banner
        banner = tk.Frame(main_area, bg="#bdc3c7", height=50)
        banner.pack(fill=tk.X)
        ttk.Label(banner, text="\u26A1 Select any signal below to run the MetaModel classification", font=("Segoe UI", 12, "italic"), background="#bdc3c7").pack(pady=10)

        # Signal Grid Area
        self.plot_frame = tk.Frame(main_area, bg="#ecf0f1")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        for i in range(2):
            self.plot_frame.columnconfigure(i, weight=1)
            self.plot_frame.rowconfigure(i, weight=1)

        # Bottom Results Area
        results_frame = tk.Frame(main_area, bg="white", height=250, relief="flat", bd=1)
        results_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=20)
        results_frame.pack_propagate(False)

        # Results header
        ttk.Label(results_frame, text="Diagnostic Output", font=("Segoe UI", 14, "bold"), background="white").pack(anchor="nw", padx=20, pady=(10, 0))
        
        res_split = tk.Frame(results_frame, bg="white")
        res_split.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Probabilities
        prob_frame = tk.Frame(res_split, bg="white")
        prob_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.lbl_diagnosis = tk.Label(prob_frame, text="Awaiting Selection...", font=("Segoe UI", 20, "bold"), bg="white", fg="#7f8c8d")
        self.lbl_diagnosis.pack(anchor="w", pady=(10, 5))
        
        self.lbl_probs = tk.Label(prob_frame, text="", font=("Segoe UI", 12), bg="white")
        self.lbl_probs.pack(anchor="w")

        # History Treeview
        hist_frame = tk.Frame(res_split, bg="white")
        hist_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        columns = ("Time", "True", "Predicted", "Conf")
        self.tree = ttk.Treeview(hist_frame, columns=columns, show="headings", height=6)
        self.tree.heading("Time", text="Time")
        self.tree.heading("True", text="True Label")
        self.tree.heading("Predicted", text="Predicted")
        self.tree.heading("Conf", text="Confidence")
        self.tree.column("Time", width=80)
        self.tree.column("True", width=120)
        self.tree.column("Predicted", width=120)
        self.tree.column("Conf", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
    def get_selected_channel_idx(self):
        val = self.channel_var.get()
        return int(val.split(" ")[1]) # extracts the number

    def load_random_signals(self):
        self.lbl_diagnosis.config(text="Fetching signals...", fg="#7f8c8d")
        self.lbl_probs.config(text="")
        self.update()
        
        self.current_signals = []
        selected_files = random.sample(self.valid_files, 4)
        
        for i, filepath in enumerate(selected_files):
            df = read_eeg_file(filepath)
            raw_matrix = df.pivot(index='sensor_position', columns='sample_num', values='sensor_value').values
            if raw_matrix.shape != (64, 256):
                continue
                
            folder_name = os.path.basename(os.path.dirname(filepath))
            actual_label = "Alcoholic" if 'co2a' in folder_name else "Control"
            filtered_matrix = apply_bulk_bandpass(raw_matrix)
            
            self.current_signals.append({
                'id': i,
                'matrix': filtered_matrix,
                'actual_label': actual_label,
                'filepath': filepath
            })
            
        self.replot_signals()
        self.lbl_diagnosis.config(text="Select a plot above to test.")

    def replot_signals(self):
        # Clear old plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        ch_idx = self.get_selected_channel_idx()
        
        for i, sig_data in enumerate(self.current_signals):
            fig, ax = plt.subplots(figsize=(5, 3), dpi=90)
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8f9fa')
            
            signal = sig_data['matrix'][ch_idx, :]
            ax.plot(signal, color='#2c3e50', linewidth=1.2)
            
            ax.set_title(f"Trial {i+1} | Channel {ch_idx}", fontsize=10, fontweight='bold', color='#34495e')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.margins(x=0.01)
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.config(cursor="hand2")
            
            row = i // 2
            col = i % 2
            widget.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
            
            # Hover bindings
            widget.bind("<Enter>", lambda e, w=widget: w.config(bg="#dff9fb"))
            widget.bind("<Leave>", lambda e, w=widget: w.config(bg="#ecf0f1"))
            
            # Click binding
            widget.bind("<Button-1>", lambda e, idx=i: self.run_classification(idx))
            
    def run_classification(self, idx):
        if idx >= len(self.current_signals): return
        
        sig_data = self.current_signals[idx]
        matrix = sig_data['matrix']
        true_label = sig_data['actual_label']
        
        self.lbl_diagnosis.config(text="Analyzing spatial-temporal patterns...", fg="#f39c12")
        self.update()
        
        input_tensor = torch.FloatTensor(matrix.copy()).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            prob_control = probs[0].item() * 100
            prob_alc = probs[1].item() * 100
            pred_idx = torch.argmax(probs, dim=0).item()
            
        pred_label = "Alcoholic" if pred_idx == 1 else "Control"
        confidence = prob_alc if pred_idx == 1 else prob_control
        
        is_correct = (pred_label == true_label)
        self.total_tests += 1
        if is_correct: self.correct_preds += 1
        
        # Update Stats
        self.lbl_total.config(text=f"Tested: {self.total_tests}")
        acc = (self.correct_preds / self.total_tests) * 100
        self.lbl_acc.config(text=f"Accuracy: {acc:.1f}%")
        
        # Update UI
        color = "#27ae60" if is_correct else "#e74c3c"
        result_str = f"PREDICTION: {pred_label.upper()} ({confidence:.1f}% confidence)"
        self.lbl_diagnosis.config(text=result_str, fg=color)
        
        detail_str = f"\u25B6 True Ground Truth: {true_label}\n"
        detail_str += f"\u25B6 Output Probability: Alcoholic [{prob_alc:.1f}%]  |  Control [{prob_control:.1f}%]"
        self.lbl_probs.config(text=detail_str)
        
        # Update History
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.tree.insert("", 0, values=(time_str, true_label, pred_label, f"{confidence:.1f}%"))

# ==========================================
# MAIN LAUNCHER
# ==========================================
def main():
    # 1. Setup UI
    setup = SetupWindow()
    setup.mainloop()
    
    if not setup.user_data:
        print("Setup aborted.")
        return
        
    # 2. Check dependencies
    test_dir = r"C:\Users\Visha\Desktop\Internshhip\SMNI_CMI_TEST"
    if not os.path.exists(test_dir):
        messagebox.showerror("System Error", f"Test dataset not found at:\n{test_dir}")
        return
        
    valid_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.gz') and not file.endswith('.tar.gz'):
                valid_files.append(os.path.join(root, file))
                
    if not valid_files:
        messagebox.showerror("System Error", "No raw EEG files (.gz) found in the test directory.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TripleLayer_MetaModel(num_classes=2).to(device)
    
    model_path = r"C:\Users\Visha\Desktop\Internshhip\best_modal.pth"
    if not os.path.exists(model_path):
        messagebox.showerror("System Error", f"Model weights missing:\n{model_path}")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        messagebox.showerror("System Error", f"Failed to load Triple-Layer MetaModel:\n{str(e)}")
        return
        
    # 3. Main Dashboard
    dashboard = EEGTestDashboard(setup.user_data, valid_files, model, device)
    dashboard.mainloop()

if __name__ == "__main__":
    main()
