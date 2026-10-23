"""
Comprehensive Ablation Study for EEG MetaModel
Tests different architectural configurations to understand component contributions
Includes statistical analysis between different configurations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import copy
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from improved_data_loader import EEGDataLoader
from improved_eeg_model import (
    ImprovedTripleLayerMetaModel, 
    LightweightMetaModel,
    ImprovedLayer1,
    ImprovedLayer2,
    ImprovedLayer3,
    count_parameters
)


class AblationModel(nn.Module):
    """
    Base class for ablation study models with different configurations
    """
    
    def __init__(self, config):
        super(AblationModel, self).__init__()
        self.config = config
        
        # Layer 1: Spatio-temporal
        if config.get('use_layer1', True):
            self.layer1 = ImprovedLayer1()
            l1_output_dim = 128
        else:
            self.layer1 = None
            l1_output_dim = 0
        
        # Layer 2: Frequency-segregated
        if config.get('use_layer2', True):
            self.layer2 = ImprovedLayer2()
            l2_output_dim = 128
        else:
            self.layer2 = None
            l2_output_dim = 0
        
        # Layer 3: Meta-classifier
        input_dim = l1_output_dim + l2_output_dim
        if config.get('use_stats', True):
            # Add statistical features
            input_dim += 64  # Statistical features
            self.stats_extractor = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(inplace=True)
            )
        else:
            self.stats_extractor = None
        
        if config.get('use_layer3', True):
            self.layer3 = ImprovedLayer3(input_dim=input_dim, num_classes=2)
        else:
            # Simple classifier if layer3 is disabled
            self.layer3 = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )
    
    def forward(self, x):
        features_list = []
        
        # Layer 1
        if self.layer1 is not None:
            feat1 = self.layer1(x)
            features_list.append(feat1)
        
        # Layer 2
        if self.layer2 is not None:
            feat2 = self.layer2(x)
            features_list.append(feat2)
        
        # Statistical features
        if self.stats_extractor is not None:
            stats = torch.cat([x.mean(dim=2), x.std(dim=2), x.max(dim=2)[0], x.min(dim=2)[0]], dim=1)
            stats = self.stats_extractor(stats)
            features_list.append(stats)
        
        # Combine features
        if len(features_list) > 0:
            combined = torch.cat(features_list, dim=1)
        else:
            # Fallback: use flattened input with projection
            combined = x.reshape(x.size(0), -1)
            # Project to reasonable dimension
            if not hasattr(self, 'flatten_projection'):
                self.flatten_projection = nn.Sequential(
                    nn.Linear(combined.size(1), 256),
                    nn.ReLU(inplace=True)
                )
            combined = self.flatten_projection(combined)
        
        # Classification
        logits = self.layer3(combined)
        
        return logits


class AblationStudy:
    """
    Comprehensive ablation study framework
    """
    
    def __init__(self, device=None, random_seed=42):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"Using device: {self.device}")
    
    def define_ablation_configs(self):
        """
        Define different ablation configurations
        """
        configs = {
            'Full_Model': {
                'use_layer1': True,
                'use_layer2': True,
                'use_layer3': True,
                'use_stats': True,
                'description': 'Complete Triple-Layer MetaModel'
            },
            'No_Layer1': {
                'use_layer1': False,
                'use_layer2': True,
                'use_layer3': True,
                'use_stats': True,
                'description': 'Without spatio-temporal pathway'
            },
            'No_Layer2': {
                'use_layer1': True,
                'use_layer2': False,
                'use_layer3': True,
                'use_stats': True,
                'description': 'Without frequency-segregated pathway'
            },
            'No_Layer3': {
                'use_layer1': True,
                'use_layer2': True,
                'use_layer3': False,
                'use_stats': True,
                'description': 'Without enhanced meta-classifier'
            },
            'No_Stats': {
                'use_layer1': True,
                'use_layer2': True,
                'use_layer3': True,
                'use_stats': False,
                'description': 'Without statistical features'
            },
            'Layer1_Only': {
                'use_layer1': True,
                'use_layer2': False,
                'use_layer3': True,
                'use_stats': False,
                'description': 'Only spatio-temporal pathway'
            },
            'Layer2_Only': {
                'use_layer1': False,
                'use_layer2': True,
                'use_layer3': True,
                'use_stats': False,
                'description': 'Only frequency-segregated pathway'
            },
            'Minimal': {
                'use_layer1': False,
                'use_layer2': False,
                'use_layer3': False,
                'use_stats': False,
                'description': 'Minimal baseline (flattened input)'
            }
        }
        
        return configs
    
    def train_config(self, X, y, config_name, config, n_folds=10, epochs=25, batch_size=32):
        """
        Train a specific configuration
        """
        print(f"\n{'='*60}")
        print(f"Training Configuration: {config_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        # Create model
        model = AblationModel(config).to(self.device)
        num_params = count_parameters(model)
        print(f"Model parameters: {num_params:,}")
        
        # Create dataset
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{n_folds}")
            
            # Create data loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            
            # Initialize model for this fold
            model = AblationModel(config).to(self.device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # Training loop
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(epochs):
                # Train
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
                
                # Validate
                model.eval()
                val_correct = 0
                val_total = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                val_acc = 100.0 * val_correct / val_total
                
                scheduler.step()
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 5:
                    break
            
            # Calculate additional metrics
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            fold_results.append({
                'fold': fold + 1,
                'accuracy': best_val_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            print(f"  Best Val Acc: {best_val_acc:.2f}%")
        
        # Calculate statistics
        accuracies = [r['accuracy'] for r in fold_results]
        precisions = [r['precision'] for r in fold_results]
        recalls = [r['recall'] for r in fold_results]
        f1s = [r['f1'] for r in fold_results]
        
        results = {
            'config_name': config_name,
            'description': config['description'],
            'parameters': num_params,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'fold_results': fold_results
        }
        
        print(f"\nResults for {config_name}:")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
        print(f"  Mean F1: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
        
        return results
    
    def run_ablation_study(self, X, y, n_folds=10, epochs=25, batch_size=32):
        """
        Run complete ablation study
        """
        print("="*60)
        print("ABLATION STUDY")
        print("="*60)
        
        configs = self.define_ablation_configs()
        all_results = []
        
        for config_name, config in configs.items():
            try:
                results = self.train_config(X, y, config_name, config, n_folds, epochs, batch_size)
                all_results.append(results)
            except Exception as e:
                print(f"Error training {config_name}: {e}")
                continue
        
        return all_results
    
    def statistical_analysis(self, all_results):
        """
        Perform statistical analysis between configurations
        """
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Create comparison matrix
        config_names = [r['config_name'] for r in all_results]
        n_configs = len(config_names)
        
        # Pairwise t-tests
        p_values = np.ones((n_configs, n_configs))
        
        for i in range(n_configs):
            for j in range(i+1, n_configs):
                acc_i = [fold['accuracy'] for fold in all_results[i]['fold_results']]
                acc_j = [fold['accuracy'] for fold in all_results[j]['fold_results']]
                
                # Paired t-test
                t_stat, p_val = stats.ttest_rel(acc_i, acc_j)
                p_values[i, j] = p_val
                p_values[j, i] = p_val
        
        # Create p-value table
        p_value_df = pd.DataFrame(p_values, index=config_names, columns=config_names)
        
        print("\nP-value Matrix (Paired t-test):")
        print(p_value_df.round(4))
        
        # Identify significant differences
        print("\nSignificant Differences (p < 0.05):")
        for i in range(n_configs):
            for j in range(i+1, n_configs):
                if p_values[i, j] < 0.05:
                    print(f"  {config_names[i]} vs {config_names[j]}: p = {p_values[i, j]:.4f} *")
        
        return p_value_df
    
    def plot_ablation_results(self, all_results, save_path='ablation_results.png'):
        """
        Plot ablation study results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        config_names = [r['config_name'] for r in all_results]
        mean_accs = [r['mean_accuracy'] for r in all_results]
        std_accs = [r['std_accuracy'] for r in all_results]
        mean_f1s = [r['mean_f1'] for r in all_results]
        std_f1s = [r['std_f1'] for r in all_results]
        params = [r['parameters'] for r in all_results]
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(config_names))
        ax1.bar(x_pos, mean_accs, yerr=std_accs, capsize=5, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Mean Accuracy (%)')
        ax1.set_title('Accuracy Comparison Across Configurations')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        ax1.grid(True, axis='y')
        
        # Plot 2: F1-score comparison
        ax2 = axes[0, 1]
        ax2.bar(x_pos, mean_f1s, yerr=std_f1s, capsize=5, color='coral', alpha=0.7)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Mean F1-Score')
        ax2.set_title('F1-Score Comparison Across Configurations')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.grid(True, axis='y')
        
        # Plot 3: Parameters vs Accuracy
        ax3 = axes[1, 0]
        scatter = ax3.scatter(params, mean_accs, s=200, c=mean_accs, cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Number of Parameters')
        ax3.set_ylabel('Mean Accuracy (%)')
        ax3.set_title('Parameters vs Accuracy Trade-off')
        ax3.grid(True)
        plt.colorbar(scatter, ax=ax3, label='Accuracy (%)')
        
        # Add labels for each point
        for i, config_name in enumerate(config_names):
            ax3.annotate(config_name, (params[i], mean_accs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 4: Detailed metrics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create table data
        table_data = []
        for r in all_results:
            table_data.append([
                r['config_name'],
                f"{r['mean_accuracy']:.2f}±{r['std_accuracy']:.2f}",
                f"{r['mean_f1']:.4f}±{r['std_f1']:.4f}",
                f"{r['parameters']:,}"
            ])
        
        table = ax4.table(cellText=table_data, 
                         colLabels=['Config', 'Acc (%)', 'F1', 'Params'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation results plot saved to {save_path}")
        plt.close()
    
    def generate_ablation_table(self, all_results, save_path='ablation_table.csv'):
        """
        Generate ablation results table
        """
        table_data = []
        for r in all_results:
            table_data.append({
                'Configuration': r['config_name'],
                'Description': r['description'],
                'Parameters': r['parameters'],
                'Mean_Accuracy': r['mean_accuracy'],
                'Std_Accuracy': r['std_accuracy'],
                'Mean_Precision': r['mean_precision'],
                'Std_Precision': r['std_precision'],
                'Mean_Recall': r['mean_recall'],
                'Std_Recall': r['std_recall'],
                'Mean_F1': r['mean_f1'],
                'Std_F1': r['std_f1']
            })
        
        df = pd.DataFrame(table_data)
        df.to_csv(save_path, index=False)
        print(f"Ablation table saved to {save_path}")
        
        return df


def main():
    """
    Main function to run ablation study
    """
    print("="*60)
    print("ABLATION STUDY FOR EEG METAMODEL")
    print("="*60)
    
    # Configuration
    config = {
        'max_files': 200,  # Use 200 files for reasonable training time
        'n_folds': 10,  # Full 10-fold CV as required
        'epochs': 25,  # Full 25 epochs as required
        'batch_size': 32
    }
    
    # Load data
    print("\nLoading data...")
    loader = EEGDataLoader()
    X, y = loader.load_kaggle_dataset(max_files=config['max_files'])
    X = loader.preprocess_pipeline(X, filter=True, normalize=True, normalize_method='zscore')
    
    print(f"Data shape: {X.shape}")
    
    # Run ablation study
    study = AblationStudy()
    all_results = study.run_ablation_study(
        X, y,
        n_folds=config['n_folds'],
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )
    
    # Statistical analysis
    p_value_df = study.statistical_analysis(all_results)
    
    # Plot results
    study.plot_ablation_results(all_results, save_path='ablation_study_results.png')
    
    # Generate table
    results_df = study.generate_ablation_table(all_results, save_path='ablation_study_table.csv')
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()