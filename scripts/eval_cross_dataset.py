"""
Evaluates cross-dataset generalization from Henri Begleiter (UCI) to OpenNeuro ds005907.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
import pandas as pd
from src.data.uci_loader import load_uci_begleiter_dataset
from src.data.openneuro_processor import load_openneuro_cross_dataset
from src.models import get_model
from src.training.trainer import train_epoch, evaluate
from src.training.metrics import compute_all_metrics
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluating on device:", device)

# Load both cohorts
X_uci, y_uci, _ = load_uci_begleiter_dataset()
X_open, y_open, _ = load_openneuro_cross_dataset()

train_loader = DataLoader(TensorDataset(torch.tensor(X_uci), torch.tensor(y_uci)), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_open), torch.tensor(y_open)), batch_size=32, shuffle=False)

models = ["biostg", "conformer", "cnnlstm", "shallow", "eegnet"]
cross_records = []

for m_name in models:
    print(f"Training {m_name.upper()} on UCI (1,200 trials) and testing on OpenNeuro ds005907 (300 trials)...")
    m = get_model(m_name, num_channels=61, time_steps=256, use_3d_prior=False).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    
    # Train for 15 epochs
    for ep in range(15):
        _ = train_epoch(m, train_loader, opt, crit, device, scaler)
        
    yt, yp, ypr = evaluate(m, test_loader, device)
    metrics = compute_all_metrics(yt, yp, ypr)
    
    print(f"  {m_name:10s} -> OpenNeuro Acc: {metrics['accuracy']:.2f}% | Sens: {metrics['sensitivity']:.2f}% | "
          f"Spec: {metrics['specificity']:.2f}% | AUC: {metrics['roc_auc']:.2f}%")
          
    cross_records.append({
        "Model": m_name.upper(),
        "Training Cohort": "Henri Begleiter (UCI)",
        "Evaluation Cohort": "OpenNeuro ds005907",
        "Accuracy (%)": round(metrics["accuracy"], 2),
        "Sensitivity (%)": round(metrics["sensitivity"], 2),
        "Specificity (%)": round(metrics["specificity"], 2),
        "F1-Score (%)": round(metrics["f1_score"], 2),
        "ROC-AUC (%)": round(metrics["roc_auc"], 2)
    })

df_cross = pd.DataFrame(cross_records)
df_cross.to_csv("results/cross_dataset_results.csv", index=False)
print("Updated results/cross_dataset_results.csv with OpenNeuro ds005907 evaluation!")
