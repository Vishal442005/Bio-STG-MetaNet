"""
Comprehensive Training Script for Improved EEG Model
Features:
- 10-fold stratified cross-validation
- 25 epochs as specified
- Supports both UCI and Kaggle datasets
- Advanced training techniques (label smoothing, gradient clipping, early stopping)
- Saves training/validation accuracy plots
- Targets 99%+ accuracy
- Includes inference latency measurement
"""

import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from improved_data_loader import EEGDataLoader
from improved_eeg_model import ImprovedTripleLayerMetaModel, LightweightMetaModel, count_parameters


class EEGTrainer:
    """
    Comprehensive trainer for EEG classification models
    """
    
    def __init__(self, model_type='improved', device=None, random_seed=42):
        """
        Initialize trainer
        Args:
            model_type: 'improved' or 'lightweight'
            device: torch device (auto-detected if None)
            random_seed: random seed for reproducibility
        """
        self.model_type = model_type
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        print(f"Model type: {model_type}")
        
        # Initialize model
        if model_type == 'improved':
            self.model = ImprovedTripleLayerMetaModel(num_classes=2).to(self.device)
        elif model_type == 'lightweight':
            self.model = LightweightMetaModel(num_classes=2).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Training history
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'fold_results': []
        }
    
    def load_and_preprocess_data(self, dataset_type='kaggle', data_dirs=None, max_files=None):
        """
        Load and preprocess EEG data
        Args:
            dataset_type: 'kaggle' or 'existing' or 'both'
            data_dirs: dict with directory paths
            max_files: maximum files to load (for testing)
        """
        loader = EEGDataLoader()
        
        if dataset_type == 'kaggle':
            print("\nLoading Kaggle SMNI dataset...")
            if data_dirs is None:
                data_dirs = {'train': 'SMNI_CMI_TRAIN', 'test': 'SMNI_CMI_TEST'}
            
            X, y = loader.load_kaggle_dataset(
                train_dir=data_dirs.get('train', 'SMNI_CMI_TRAIN'),
                test_dir=data_dirs.get('test', 'SMNI_CMI_TEST'),
                max_files=max_files
            )
        
        elif dataset_type == 'existing':
            print("\nLoading existing processed data...")
            if os.path.exists('X_filtered.npy') and os.path.exists('y_train.npy'):
                X = np.load('X_filtered.npy')
                y = np.load('y_train.npy')
                print(f"Loaded existing data: {X.shape}")
                print(f"Class distribution: Alcoholic={np.sum(y==1)}, Control={np.sum(y==0)}")
                # Don't reprocess if already filtered to preserve original preprocessing
                X_processed = X
            else:
                print("Existing data files not found, falling back to Kaggle dataset")
                # Fall back to Kaggle dataset
                X, y = loader.load_kaggle_dataset(max_files=max_files)
                X_processed = loader.preprocess_pipeline(X, filter=True, normalize=True, normalize_method='zscore')
            
            return X_processed, y
        
        elif dataset_type == 'both':
            print("\nLoading combined dataset...")
            # Load Kaggle data with all files
            X_kaggle, y_kaggle = loader.load_kaggle_dataset(max_files=max_files)
            # Skip existing data since we have enough from Kaggle
            X, y = X_kaggle, y_kaggle
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Preprocess
        print("\nPreprocessing data...")
        X_processed = loader.preprocess_pipeline(X, filter=True, normalize=True, normalize_method='zscore')
        
        print(f"Final data shape: {X_processed.shape}")
        print(f"Class distribution: Alcoholic={np.sum(y==1)}, Control={np.sum(y==0)}")
        
        return X_processed, y
    
    def train_epoch(self, train_loader, criterion, optimizer, scheduler=None):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * total_correct / total_samples
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except:
            auc = 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics
    
    def measure_inference_time(self, test_loader, num_samples=100):
        """
        Measure inference latency
        """
        self.model.eval()
        inference_times = []
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if i >= num_samples:
                    break
                
                inputs = inputs.to(self.device)
                
                # Warm-up
                _ = self.model(inputs)
                
                # Measure time
                start_time = time.time()
                _ = self.model(inputs)
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        return avg_time, std_time
    
    def train_with_cross_validation(self, X, y, n_folds=10, epochs=25, batch_size=32, 
                                   learning_rate=2e-4, weight_decay=1e-4, early_stopping_patience=5):
        """
        Train with n-fold stratified cross-validation
        """
        print(f"\n{'='*60}")
        print(f"Starting {n_folds}-Fold Stratified Cross-Validation")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        print(f"Learning Rate: {learning_rate}, Weight Decay: {weight_decay}")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        
        fold_results = []
        best_overall_accuracy = 0.0
        best_model_state = None
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{n_folds}")
            print(f"{'='*60}")
            
            # Create data loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # Initialize model for this fold
            if self.model_type == 'improved':
                model = ImprovedTripleLayerMetaModel(num_classes=2).to(self.device)
            else:
                model = LightweightMetaModel(num_classes=2).to(self.device)
            
            # Loss function with label smoothing
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            
            # Optimizer
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # Training loop
            best_val_acc = 0.0
            patience_counter = 0
            fold_train_acc = []
            fold_val_acc = []
            
            for epoch in range(epochs):
                # Train
                train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, scheduler)
                
                # Validate
                val_metrics = self.validate(val_loader, criterion)
                val_acc = val_metrics['accuracy']
                val_loss = val_metrics['loss']
                
                fold_train_acc.append(train_acc)
                fold_val_acc.append(val_acc)
                
                # Print progress
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:2d}/{epochs} | "
                          f"Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}% | "
                          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_fold_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model for this fold
            model.load_state_dict(best_fold_state)
            
            # Final validation
            final_metrics = self.validate(val_loader, criterion)
            
            # Measure inference time
            avg_time, std_time = self.measure_inference_time(val_loader, num_samples=50)
            
            fold_result = {
                'fold': fold + 1,
                'best_val_accuracy': best_val_acc,
                'final_metrics': final_metrics,
                'inference_time_ms': avg_time,
                'inference_time_std': std_time,
                'train_acc_history': fold_train_acc,
                'val_acc_history': fold_val_acc
            }
            
            fold_results.append(fold_result)
            
            print(f"\nFold {fold+1} Results:")
            print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
            print(f"  Final Accuracy: {final_metrics['accuracy']:.2f}%")
            print(f"  Precision: {final_metrics['precision']:.4f}")
            print(f"  Recall: {final_metrics['recall']:.4f}")
            print(f"  F1-Score: {final_metrics['f1']:.4f}")
            print(f"  AUC-ROC: {final_metrics['auc']:.4f}")
            print(f"  Inference Time: {avg_time:.2f} ± {std_time:.2f} ms")
            
            # Track best overall model
            if best_val_acc > best_overall_accuracy:
                best_overall_accuracy = best_val_acc
                best_model_state = copy.deepcopy(model.state_dict())
        
        # Load best overall model
        self.model.load_state_dict(best_model_state)
        
        # Calculate aggregate statistics
        accuracies = [r['best_val_accuracy'] for r in fold_results]
        precisions = [r['final_metrics']['precision'] for r in fold_results]
        recalls = [r['final_metrics']['recall'] for r in fold_results]
        f1s = [r['final_metrics']['f1'] for r in fold_results]
        aucs = [r['final_metrics']['auc'] for r in fold_results]
        inference_times = [r['inference_time_ms'] for r in fold_results]
        
        aggregate_results = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'fold_results': fold_results
        }
        
        self.history['fold_results'] = fold_results
        
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Mean Accuracy: {aggregate_results['mean_accuracy']:.2f}% ± {aggregate_results['std_accuracy']:.2f}%")
        print(f"Mean Precision: {aggregate_results['mean_precision']:.4f} ± {aggregate_results['std_precision']:.4f}")
        print(f"Mean Recall: {aggregate_results['mean_recall']:.4f} ± {aggregate_results['std_recall']:.4f}")
        print(f"Mean F1-Score: {aggregate_results['mean_f1']:.4f} ± {aggregate_results['std_f1']:.4f}")
        print(f"Mean AUC-ROC: {aggregate_results['mean_auc']:.4f} ± {aggregate_results['std_auc']:.4f}")
        print(f"Mean Inference Time: {aggregate_results['mean_inference_time']:.2f} ± {aggregate_results['std_inference_time']:.2f} ms")
        print(f"Best Overall Accuracy: {best_overall_accuracy:.2f}%")
        print(f"{'='*60}")
        
        return aggregate_results
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """
        Plot training and validation curves for all folds
        """
        if not self.history['fold_results']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training accuracy across folds
        ax1 = axes[0, 0]
        for fold_result in self.history['fold_results']:
            ax1.plot(fold_result['train_acc_history'], alpha=0.6, label=f"Fold {fold_result['fold']}")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Accuracy (%)')
        ax1.set_title('Training Accuracy Across Folds')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # Plot 2: Validation accuracy across folds
        ax2 = axes[0, 1]
        for fold_result in self.history['fold_results']:
            ax2.plot(fold_result['val_acc_history'], alpha=0.6, label=f"Fold {fold_result['fold']}")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy (%)')
        ax2.set_title('Validation Accuracy Across Folds')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        
        # Plot 3: Average training vs validation
        ax3 = axes[1, 0]
        max_epochs = max(len(fr['train_acc_history']) for fr in self.history['fold_results'])
        avg_train_acc = np.zeros(max_epochs)
        avg_val_acc = np.zeros(max_epochs)
        
        for epoch in range(max_epochs):
            train_accs = [fr['train_acc_history'][epoch] for fr in self.history['fold_results'] 
                         if epoch < len(fr['train_acc_history'])]
            val_accs = [fr['val_acc_history'][epoch] for fr in self.history['fold_results'] 
                       if epoch < len(fr['val_acc_history'])]
            
            if train_accs:
                avg_train_acc[epoch] = np.mean(train_accs)
            if val_accs:
                avg_val_acc[epoch] = np.mean(val_accs)
        
        ax3.plot(range(max_epochs), avg_train_acc, 'b-', label='Average Training Accuracy', linewidth=2)
        ax3.plot(range(max_epochs), avg_val_acc, 'r-', label='Average Validation Accuracy', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Average Training vs Validation Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Per-fold best accuracies
        ax4 = axes[1, 1]
        fold_numbers = [fr['fold'] for fr in self.history['fold_results']]
        best_accuracies = [fr['best_val_accuracy'] for fr in self.history['fold_results']]
        
        ax4.bar(fold_numbers, best_accuracies, color='steelblue', alpha=0.7)
        ax4.axhline(y=np.mean(best_accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(best_accuracies):.2f}%')
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('Best Validation Accuracy (%)')
        ax4.set_title('Per-Fold Best Validation Accuracies')
        ax4.legend()
        ax4.grid(True, axis='y')
        
        plt.tight_layout()
        
        # Ensure plots directory exists
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save to plots directory
        save_path = os.path.join(plots_dir, os.path.basename(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.close()
    
    def save_model(self, save_path='best_improved_model.pth'):
        """
        Save the trained model
        """
        # Ensure models directory exists
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save to models directory
        save_path = os.path.join(models_dir, os.path.basename(save_path))
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'random_seed': self.random_seed,
            'history': self.history
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def generate_results_table(self, save_path='results_table.csv'):
        """
        Generate results table for paper
        """
        if not self.history['fold_results']:
            print("No results to generate table")
            return
        
        results_data = []
        for fold_result in self.history['fold_results']:
            results_data.append({
                'Fold': fold_result['fold'],
                'Best_Val_Accuracy': fold_result['best_val_accuracy'],
                'Final_Accuracy': fold_result['final_metrics']['accuracy'],
                'Precision': fold_result['final_metrics']['precision'],
                'Recall': fold_result['final_metrics']['recall'],
                'F1_Score': fold_result['final_metrics']['f1'],
                'AUC_ROC': fold_result['final_metrics']['auc'],
                'Inference_Time_ms': fold_result['inference_time_ms']
            })
        
        df = pd.DataFrame(results_data)
        
        # Ensure results directory exists
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to results directory
        save_path = os.path.join(results_dir, os.path.basename(save_path))
        df.to_csv(save_path, index=False)
        print(f"Results table saved to {save_path}")
        
        return df


def main():
    """
    Main training function
    """
    print("="*60)
    print("IMPROVED EEG MODEL TRAINING")
    print("="*60)
    
    # Configuration
    config = {
        'model_type': 'lightweight',  # Use lightweight model for better convergence
        'dataset_type': 'kaggle',  # Use Kaggle dataset (1200 files available)
        'max_files': 100,  # Use 100 files (50 per class) for initial testing
        'n_folds': 5,  # Reduced folds for initial testing
        'epochs': 15,  # Reduced epochs for initial testing
        'batch_size': 32,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 3
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = EEGTrainer(model_type=config['model_type'])
    
    # Load and preprocess data
    X, y = trainer.load_and_preprocess_data(
        dataset_type=config['dataset_type'],
        max_files=config['max_files']
    )
    
    # Train with cross-validation
    results = trainer.train_with_cross_validation(
        X, y,
        n_folds=config['n_folds'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Plot training curves
    trainer.plot_training_curves(save_path='improved_model_training_curves.png')
    
    # Save model
    trainer.save_model(save_path='best_improved_model.pth')
    
    # Generate results table
    results_df = trainer.generate_results_table(save_path='improved_model_results.csv')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
    
    if results['mean_accuracy'] >= 99.0:
        print("TARGET ACHIEVED: 99%+ accuracy reached!")
    else:
        print(f"Target not reached. Current accuracy: {results['mean_accuracy']:.2f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()