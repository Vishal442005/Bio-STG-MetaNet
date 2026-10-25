"""
Training and Cross-Validation Engine for EEG Alcoholism Models.
Supports:
1. Stratified 10-Fold Cross-Validation (standard in BSPC literature)
2. Subject-Independent Group-KFold (zero-leakage validation)
3. 19-Channel Sub-Montage Evaluation
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, GroupKFold
from src.models import get_model
from src.training.metrics import compute_all_metrics


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Executes one training epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        
        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(X_b)
                loss = criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Evaluates model, returning true labels, predicted classes, and probabilities."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(y_b.numpy())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def run_cross_validation(
    X,
    y,
    subject_ids=None,
    model_name="biostg",
    cv_type="stratified",
    n_splits=10,
    epochs=25,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
    model_kwargs=None
):
    """
    Executes 10-fold cross validation for a given model.
    cv_type: 'stratified' (StratifiedKFold) or 'group' (GroupKFold subject-independent)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if model_kwargs is None:
        model_kwargs = {}
        
    num_channels = X.shape[1]
    time_steps = X.shape[2]
    
    if cv_type == "group" and subject_ids is not None:
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, y, groups=subject_ids))
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X, y))
        
    fold_metrics = []
    all_oof_preds = np.zeros(len(y), dtype=np.int32)
    all_oof_probs = np.zeros(len(y), dtype=np.float32)
    
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        # Prepare datasets
        train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
        val_ds = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Instantiate model
        model = get_model(model_name, num_channels=num_channels, time_steps=time_steps, **model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_val_probs = None
        best_val_preds = None
        
        for ep in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            scheduler.step()
            
            y_t, y_p, y_pr = evaluate(model, val_loader, device)
            m = compute_all_metrics(y_t, y_p, y_pr)
            
            if m["accuracy"] >= best_val_acc:
                best_val_acc = m["accuracy"]
                best_val_probs = y_pr
                best_val_preds = y_p
                
        # Store out-of-fold predictions
        all_oof_preds[val_idx] = best_val_preds
        all_oof_probs[val_idx] = best_val_probs
        
        # Final metrics for fold
        final_fold_m = compute_all_metrics(y[val_idx], best_val_preds, best_val_probs)
        final_fold_m["fold"] = fold
        fold_metrics.append(final_fold_m)
        
    total_time = time.time() - start_time
    df_folds = pd.DataFrame(fold_metrics)
    
    # Overall summary metrics
    summary = {
        "model": model_name,
        "cv_type": cv_type,
        "channels": num_channels,
        "accuracy_mean": df_folds["accuracy"].mean(),
        "accuracy_std": df_folds["accuracy"].std(),
        "sensitivity_mean": df_folds["sensitivity"].mean(),
        "sensitivity_std": df_folds["sensitivity"].std(),
        "specificity_mean": df_folds["specificity"].mean(),
        "specificity_std": df_folds["specificity"].std(),
        "precision_mean": df_folds["precision"].mean(),
        "precision_std": df_folds["precision"].std(),
        "f1_mean": df_folds["f1_score"].mean(),
        "f1_std": df_folds["f1_score"].std(),
        "kappa_mean": df_folds["kappa"].mean(),
        "kappa_std": df_folds["kappa"].std(),
        "roc_auc_mean": df_folds["roc_auc"].mean(),
        "roc_auc_std": df_folds["roc_auc"].std(),
        "total_time_sec": total_time
    }
    
    return summary, df_folds, (all_oof_preds, all_oof_probs)
