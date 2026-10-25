import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
from src.data.uci_loader import load_uci_begleiter_dataset
from src.models import get_model
from src.training.trainer import train_epoch, evaluate
from src.training.metrics import compute_all_metrics
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

X, y, subs = load_uci_begleiter_dataset()
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(kf.split(X, y))

train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
val_ds = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Test different learning rates and epochs
for lr in [1e-3, 5e-4, 3e-4]:
    print(f"\nTesting lr={lr}...")
    model = get_model("biostg", num_channels=61, time_steps=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
    crit = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    
    best_acc = 0
    for ep in range(35):
        loss = train_epoch(model, train_loader, opt, crit, device, scaler)
        scheduler.step()
        yt, yp, ypr = evaluate(model, val_loader, device)
        m = compute_all_metrics(yt, yp, ypr)
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            
        if (ep + 1) % 10 == 0 or ep == 34:
            print(f"  Epoch {ep+1:2d} -> Loss: {loss:.4f} | Val Acc: {m['accuracy']:.2f}% | Best: {best_acc:.2f}%")
            
    print(f"Final Best Fold Acc for lr={lr}: {best_acc:.2f}%")
