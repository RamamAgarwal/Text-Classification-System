"""
Evaluation and Visualization Module
Creates visualizations for model performance comparison
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_model_comparison(results_dict, save_path='results/model_comparison.png'):
    """
    Create bar chart comparing all models
    
    Args:
        results_dict: Dictionary of model results
        save_path: Path to save the plot
    """
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Prepare data
    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Score': results_dict[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='Score', hue='Metric', ax=ax)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Metrics', loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()


def plot_confusion_matrices(results_dict, save_path='results/confusion_matrices.png'):
    """
    Create confusion matrices for all models
    
    Args:
        results_dict: Dictionary of model results
        save_path: Path to save the plot
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        y_true = results['true_labels']
        y_pred = results['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   ax=axes[idx], cbar_kws={'label': 'Normalized Count'})
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices to {save_path}")
    plt.close()


def create_results_table(results_dict, save_path='results/results_table.csv'):
    """
    Create a CSV table with all model results
    
    Args:
        results_dict: Dictionary of model results
        save_path: Path to save the CSV
    """
    data = []
    for model_name, results in results_dict.items():
        data.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1-Score': f"{results['f1_score']:.4f}"
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('F1-Score', ascending=False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved results table to {save_path}")
    print("\nResults Summary:")
    print(df.to_string(index=False))


def generate_all_visualizations(results_dict):
    """
    Generate all visualizations and save results
    
    Args:
        results_dict: Dictionary of model results
    """
    print("\nGenerating visualizations...")
    plot_model_comparison(results_dict)
    plot_confusion_matrices(results_dict)
    create_results_table(results_dict)
    print("\nAll visualizations generated successfully!")


def print_detailed_report(results_dict):
    """
    Print detailed classification reports for all models
    
    Args:
        results_dict: Dictionary of model results
    """
    for model_name, results in results_dict.items():
        print(f"\n{'='*60}")
        print(f"Detailed Report for {model_name}")
        print(f"{'='*60}")
        print(classification_report(
            results['true_labels'],
            results['predictions'],
            zero_division=0
        ))


if __name__ == '__main__':
    # This would be called after training models
    # Example usage:
    # from train_models import ModelTrainer
    # trainer = ModelTrainer(...)
    # trainer.train_all_models(...)
    # generate_all_visualizations(trainer.results)
    pass
