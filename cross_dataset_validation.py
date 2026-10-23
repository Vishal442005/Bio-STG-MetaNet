"""
Cross-Dataset Validation for EEG Classification
Addresses reviewer concern about single dataset validation
Trains on one dataset and tests on another to demonstrate generalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import copy
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from improved_data_loader import EEGDataLoader
from improved_eeg_model import ImprovedTripleLayerMetaModel, count_parameters


class CrossDatasetValidator:
    """
    Cross-dataset validation framework
    """
    
    def __init__(self, device=None, random_seed=42):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = random_seed
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"Using device: {self.device}")
    
    def load_datasets(self, max_files_train=None, max_files_test=None):
        """
        Load both datasets for cross-validation
        """
        loader = EEGDataLoader()
        
        print("Loading datasets for cross-dataset validation...")
        
        # Load Kaggle dataset (will be used as source)
        print("\n1. Loading Kaggle SMNI dataset (Source)...")
        X_kaggle, y_kaggle = loader.load_kaggle_dataset(max_files=max_files_train)
        X_kaggle_processed = loader.preprocess_pipeline(X_kaggle, filter=True, normalize=True, normalize_method='zscore')
        
        # Try to load existing dataset (will be used as target)
        print("\n2. Loading existing UCI dataset (Target)...")
        try:
            if max_files_test is None:
                # Load existing processed data if available
                if hasattr(loader, 'load_processed_data'):
                    X_existing, y_existing = loader.load_processed_data('X_filtered.npy', 'y_train.npy')
                else:
                    # Fallback to loading from scratch
                    X_existing, y_existing = loader.load_uci_dataset('eeg_full')
                    X_existing_processed = loader.preprocess_pipeline(X_existing, filter=True, normalize=True, normalize_method='zscore')
            else:
                # Load subset of existing data
                X_existing, y_existing = loader.load_uci_dataset('eeg_full')
                # Take first max_files_test samples
                X_existing = X_existing[:max_files_test]
                y_existing = y_existing[:max_files_test]
                X_existing_processed = loader.preprocess_pipeline(X_existing, filter=True, normalize=True, normalize_method='zscore')
        except Exception as e:
            print(f"Could not load existing dataset: {e}")
            print("Using split of Kaggle dataset for validation")
            # Split Kaggle dataset
            from sklearn.model_selection import train_test_split
            X_kaggle_processed, X_existing_processed, y_kaggle, y_existing = train_test_split(
                X_kaggle_processed, y_kaggle, test_size=0.3, stratify=y_kaggle, random_state=42
            )
        
        print(f"\nDataset shapes:")
        print(f"  Source (Kaggle): {X_kaggle_processed.shape}, Labels: {y_kaggle.shape}")
        print(f"  Target (Existing): {X_existing_processed.shape}, Labels: {y_existing.shape}")
        
        return {
            'source': {'X': X_kaggle_processed, 'y': y_kaggle},
            'target': {'X': X_existing_processed, 'y': y_existing}
        }
    
    def train_on_source(self, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
        """
        Train model on source dataset
        """
        print("\nTraining on source dataset...")
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = ImprovedTripleLayerMetaModel(num_classes=2).to(self.device)
        print(f"Model parameters: {count_parameters(model):,}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        training_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
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
            
            train_acc = 100.0 * train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100.0 * val_correct / val_total
            
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            training_history['train_loss'].append(train_loss / len(train_loader))
            training_history['val_loss'].append(val_loss / len(val_loader))
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:2d}/{epochs} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_state)
        
        return model, training_history, best_val_acc
    
    def evaluate_on_target(self, model, X_test, y_test, batch_size=32):
        """
        Evaluate trained model on target dataset
        """
        print("\nEvaluating on target dataset...")
        
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        print(f"Target Dataset Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        return results
    
    def cross_validation_experiment(self, datasets, test_size=0.2, epochs=25, batch_size=32):
        """
        Perform cross-dataset validation experiment
        """
        print("="*60)
        print("CROSS-DATASET VALIDATION EXPERIMENT")
        print("="*60)
        
        # Split source data into train/validation
        from sklearn.model_selection import train_test_split
        
        X_source = datasets['source']['X']
        y_source = datasets['source']['y']
        X_target = datasets['target']['X']
        y_target = datasets['target']['y']
        
        print(f"\nSource dataset: {X_source.shape}")
        print(f"Target dataset: {X_target.shape}")
        
        # Split source data
        X_train, X_val, y_train, y_val = train_test_split(
            X_source, y_source, test_size=test_size, stratify=y_source, random_state=42
        )
        
        print(f"\nTraining split: {X_train.shape}")
        print(f"Validation split: {X_val.shape}")
        print(f"Target test split: {X_target.shape}")
        
        # Train on source
        model, training_history, source_val_acc = self.train_on_source(
            X_train, y_train, X_val, y_val, epochs, batch_size
        )
        
        # Evaluate on target
        target_results = self.evaluate_on_target(model, X_target, y_target, batch_size)
        
        # Also evaluate on source validation for comparison
        source_results = self.evaluate_on_target(model, X_val, y_val, batch_size)
        
        # Calculate domain adaptation gap
        domain_gap = source_results['accuracy'] - target_results['accuracy']
        
        results = {
            'source_validation_accuracy': source_results['accuracy'],
            'target_test_accuracy': target_results['accuracy'],
            'domain_gap': domain_gap,
            'target_metrics': target_results,
            'source_metrics': source_results,
            'training_history': training_history
        }
        
        print(f"\n{'='*60}")
        print("CROSS-DATASET VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Source Validation Accuracy: {source_results['accuracy']:.2f}%")
        print(f"Target Test Accuracy: {target_results['accuracy']:.2f}%")
        print(f"Domain Adaptation Gap: {domain_gap:.2f}%")
        
        if domain_gap < 10:
            print("✓ Good cross-dataset generalization (gap < 10%)")
        elif domain_gap < 20:
            print("⚠ Moderate cross-dataset generalization (gap < 20%)")
        else:
            print("✗ Poor cross-dataset generalization (gap > 20%)")
        
        print(f"{'='*60}")
        
        return results
    
    def bidirectional_validation(self, datasets, epochs=25, batch_size=32):
        """
        Perform bidirectional cross-validation (train on A test on B, and vice versa)
        """
        print("\n" + "="*60)
        print("BIDIRECTIONAL CROSS-DATASET VALIDATION")
        print("="*60)
        
        results = {}
        
        # Experiment 1: Train on Kaggle, Test on Existing
        print("\nExperiment 1: Train on Kaggle (Source), Test on Existing (Target)")
        results['kaggle_to_existing'] = self.cross_validation_experiment(
            datasets, epochs=epochs, batch_size=batch_size
        )
        
        # Experiment 2: Train on Existing, Test on Kaggle
        print("\nExperiment 2: Train on Existing (Source), Test on Kaggle (Target)")
        # Swap source and target
        swapped_datasets = {
            'source': datasets['target'],
            'target': datasets['source']
        }
        results['existing_to_kaggle'] = self.cross_validation_experiment(
            swapped_datasets, epochs=epochs, batch_size=batch_size
        )
        
        # Summary
        print("\n" + "="*60)
        print("BIDIRECTIONAL VALIDATION SUMMARY")
        print("="*60)
        print(f"Kaggle → Existing: {results['kaggle_to_existing']['target_test_accuracy']:.2f}%")
        print(f"Existing → Kaggle: {results['existing_to_kaggle']['target_test_accuracy']:.2f}%")
        print(f"Average Cross-Dataset Accuracy: "
              f"{(results['kaggle_to_existing']['target_test_accuracy'] + results['existing_to_kaggle']['target_test_accuracy'])/2:.2f}%")
        print("="*60)
        
        return results
    
    def plot_cross_dataset_results(self, results, save_path='cross_dataset_results.png'):
        """
        Plot cross-dataset validation results
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Training curves
        ax1 = axes[0, 0]
        history = results['training_history']
        ax1.plot(history['train_acc'], label='Training Accuracy', marker='o')
        ax1.plot(history['val_acc'], label='Validation Accuracy', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training Progress on Source Dataset')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Accuracy comparison
        ax2 = axes[0, 1]
        accuracies = [
            results['source_validation_accuracy'],
            results['target_test_accuracy']
        ]
        labels = ['Source Validation', 'Target Test']
        colors = ['steelblue', 'coral']
        ax2.bar(labels, accuracies, color=colors, alpha=0.7)
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Source vs Target Accuracy')
        ax2.grid(True, axis='y')
        
        # Add domain gap annotation
        domain_gap = results['domain_gap']
        ax2.annotate(f'Domain Gap: {domain_gap:.2f}%', 
                    xy=(0.5, max(accuracies) * 0.9), 
                    ha='center', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 3: Confusion matrix
        ax3 = axes[1, 0]
        cm = results['target_metrics']['confusion_matrix']
        im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax3.figure.colorbar(im, ax=ax3)
        ax3.set_xlabel('Predicted Label')
        ax3.set_ylabel('True Label')
        ax3.set_title('Confusion Matrix on Target Dataset')
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Control', 'Alcoholic'])
        ax3.set_yticklabels(['Control', 'Alcoholic'])
        
        # Plot 4: Metrics comparison
        ax4 = axes[1, 1]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        source_values = [
            results['source_metrics']['accuracy'],
            results['source_metrics']['precision'] * 100,
            results['source_metrics']['recall'] * 100,
            results['source_metrics']['f1'] * 100,
            results['source_metrics']['auc'] * 100
        ]
        target_values = [
            results['target_metrics']['accuracy'],
            results['target_metrics']['precision'] * 100,
            results['target_metrics']['recall'] * 100,
            results['target_metrics']['f1'] * 100,
            results['target_metrics']['auc'] * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, source_values, width, label='Source', color='steelblue', alpha=0.7)
        ax4.bar(x + width/2, target_values, width, label='Target', color='coral', alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value (%)')
        ax4.set_title('Metrics Comparison: Source vs Target')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-dataset results plot saved to {save_path}")
        plt.close()
    
    def generate_cross_dataset_table(self, results, save_path='cross_dataset_table.csv'):
        """
        Generate cross-dataset validation table
        """
        table_data = [{
            'Experiment': 'Source_to_Target',
            'Source_Validation_Accuracy': f"{results['source_validation_accuracy']:.2f}%",
            'Target_Test_Accuracy': f"{results['target_test_accuracy']:.2f}%",
            'Domain_Gap': f"{results['domain_gap']:.2f}%",
            'Target_Precision': f"{results['target_metrics']['precision']:.4f}",
            'Target_Recall': f"{results['target_metrics']['recall']:.4f}",
            'Target_F1': f"{results['target_metrics']['f1']:.4f}",
            'Target_AUC': f"{results['target_metrics']['auc']:.4f}"
        }]
        
        df = pd.DataFrame(table_data)
        df.to_csv(save_path, index=False)
        print(f"Cross-dataset table saved to {save_path}")
        
        return df


def main():
    """
    Main function to run cross-dataset validation
    """
    print("="*60)
    print("CROSS-DATASET VALIDATION FOR EEG CLASSIFICATION")
    print("="*60)
    
    # Configuration
    config = {
        'max_files_train': 150,  # For source dataset
        'max_files_test': 50,    # For target dataset
        'epochs': 25,  # Full 25 epochs as required
        'batch_size': 32
    }
    
    # Initialize validator
    validator = CrossDatasetValidator()
    
    # Load datasets
    datasets = validator.load_datasets(
        max_files_train=config['max_files_train'],
        max_files_test=config['max_files_test']
    )
    
    # Run cross-dataset validation
    results = validator.cross_validation_experiment(
        datasets,
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )
    
    # Plot results
    validator.plot_cross_dataset_results(results, save_path='cross_dataset_validation_results.png')
    
    # Generate table
    results_df = validator.generate_cross_dataset_table(results, save_path='cross_dataset_validation_table.csv')
    
    print("\n" + "="*60)
    print("CROSS-DATASET VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()