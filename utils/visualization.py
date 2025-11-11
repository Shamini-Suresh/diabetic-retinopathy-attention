#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Visualization utilities for training history, metrics, and attention maps
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm


def plot_training_history(fold_histories, save_path=None):
    """
    Plot mean ± std training history across all folds

    Args:
        fold_histories: List of fold history dictionaries
        save_path: Path to save the plot
    """
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc',
               'val_kappa', 'val_weighted_kappa', 'val_f1_macro', 'val_f1_weighted']
    titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy',
              'Validation Kappa (Quadratic)', 'Validation Kappa (Linear)', 
              'F1 Score (Macro)', 'F1 Score (Weighted)']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Training History Across All Folds', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        if fold_histories:
            max_epochs = max(len(fold_history.get(metric, [])) for fold_history in fold_histories)

            epoch_values = []
            for epoch in range(max_epochs):
                values_at_epoch = []
                for fold_history in fold_histories:
                    if metric in fold_history and epoch < len(fold_history[metric]):
                        values_at_epoch.append(fold_history[metric][epoch])
                epoch_values.append(values_at_epoch)

            means, stds, valid_epochs = [], [], []

            for epoch, values in enumerate(epoch_values):
                if len(values) > 0:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                    valid_epochs.append(epoch + 1)

            if means:
                means = np.array(means)
                stds = np.array(stds)

                ax.plot(valid_epochs, means, 'b-', linewidth=3, label='Mean', alpha=0.9)
                ax.fill_between(valid_epochs, means - stds, means + stds,
                               alpha=0.3, color='blue', label='± 1 Std')

                if 'loss' in metric:
                    best_idx = np.argmin(means)
                    best_value = means[best_idx]
                else:
                    best_idx = np.argmax(means)
                    best_value = means[best_idx]

                ax.plot(valid_epochs[best_idx], best_value, 'ro', markersize=10,
                       label=f'Best: {best_value:.4f}')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    plt.show()


def plot_per_class_metrics(results, class_names=None, save_path=None):
    """
    Plot detailed per-class metrics visualization

    Args:
        results: Evaluation results dictionary
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')

    # F1 scores
    ax1 = axes[0, 0]
    bars1 = ax1.bar(class_names, results['per_class_f1'], alpha=0.7, color='skyblue')
    ax1.set_title('Per-Class F1 Scores', fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars1, results['per_class_f1']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # AUC scores
    ax2 = axes[0, 1]
    valid_auc = [auc if not np.isnan(auc) else 0 for auc in results['per_class_auc']]
    bars2 = ax2.bar(class_names, valid_auc, alpha=0.7, color='lightcoral')
    ax2.set_title('Per-Class AUC Scores', fontweight='bold')
    ax2.set_ylabel('AUC Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars2, valid_auc):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Precision vs Recall
    ax3 = axes[1, 0]
    x_pos = np.arange(len(class_names))
    width = 0.35

    bars3a = ax3.bar(x_pos - width/2, results['per_class_precision'], width,
                     label='Precision', alpha=0.7, color='gold')
    bars3b = ax3.bar(x_pos + width/2, results['per_class_recall'], width,
                     label='Recall', alpha=0.7, color='lightgreen')

    ax3.set_title('Per-Class Precision vs Recall', fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.legend()

    # Class distribution
    ax4 = axes[1, 1]
    bars4 = ax4.bar(class_names, results['per_class_support'], alpha=0.7, color='mediumpurple')
    ax4.set_title('Class Distribution (Support)', fontweight='bold')
    ax4.set_ylabel('Number of Samples')
    ax4.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars4, results['per_class_support']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(results['per_class_support']) * 0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class metrics saved to {save_path}")
    plt.show()


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot confusion matrix with normalized values

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    plt.figure(figsize=(10, 8))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'})

    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Add counts
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                    ha='center', va='center', fontsize=10, color='red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_confidence_intervals(results, save_path=None):
    """
    Plot confidence intervals for main metrics

    Args:
        results: Evaluation results dictionary
        save_path: Path to save the plot
    """
    metrics = ['Accuracy', 'Quadratic Kappa', 'F1 Macro']
    values = [results['accuracy'], results['quadratic_kappa'], results['f1_macro']]
    ci_lower = [results['accuracy_ci'][0], results['kappa_ci'][0], results['f1_macro_ci'][0]]
    ci_upper = [results['accuracy_ci'][1], results['kappa_ci'][1], results['f1_macro_ci'][1]]

    errors_lower = [val - lower for val, lower in zip(values, ci_lower)]
    errors_upper = [upper - val for val, upper in zip(values, ci_upper)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bars = ax.bar(metrics, values, yerr=[errors_lower, errors_upper],
                  capsize=10, alpha=0.7, color=['skyblue', 'lightcoral', 'gold'])

    ax.set_title('Main Metrics with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)

    for i, (bar, val, lower, upper) in enumerate(zip(bars, values, ci_lower, ci_upper)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}\n[{lower:.4f}, {upper:.4f}]',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confidence intervals saved to {save_path}")
    plt.show()


def visualize_attention_maps(model, data_loader, device, num_samples=4, save_path=None):
    """
    Visualize attention maps from the model

    Args:
        model: Trained model with attention mechanism
        data_loader: DataLoader for samples
        device: torch.device
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    if not hasattr(model, 'get_attention_maps') or not model.use_attention:
        print("Model doesn't have attention mechanism")
        return

    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    sample_count = 0
    with torch.no_grad():
        for images, labels, img_names in data_loader:
            if sample_count >= num_samples:
                break

            images = images.to(device)

            # Forward pass to generate attention maps
            outputs = model(images)
            attention_maps = model.get_attention_maps()

            for i in range(min(images.size(0), num_samples - sample_count)):
                # Original image
                img = images[i].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                      torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)

                axes[sample_count, 0].imshow(img.permute(1, 2, 0))
                axes[sample_count, 0].set_title(
                    f'Original\nLabel: {labels[i].item() if labels[i] != -1 else "Unknown"}',
                    fontsize=10
                )
                axes[sample_count, 0].axis('off')

                # Show attention maps
                att_idx = 1
                for key, att_map in attention_maps.items():
                    if att_idx >= 4:
                        break
                    if 'spatial_attention' in key:
                        att = att_map[i, 0].cpu().numpy()
                        im = axes[sample_count, att_idx].imshow(att, cmap='jet', alpha=0.7)
                        axes[sample_count, att_idx].set_title(
                            f'Attention {key.split("_")[-1]}',
                            fontsize=10
                        )
                        axes[sample_count, att_idx].axis('off')
                        plt.colorbar(im, ax=axes[sample_count, att_idx], 
                                   fraction=0.046, pad=0.04)
                        att_idx += 1

                # Fill remaining slots
                while att_idx < 4:
                    axes[sample_count, att_idx].axis('off')
                    att_idx += 1

                sample_count += 1
                if sample_count >= num_samples:
                    break

    plt.suptitle('Attention Maps Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention maps saved to {save_path}")
    plt.show()


def plot_fold_comparison(fold_results, save_path=None):
    """
    Create comparison of key metrics across folds

    Args:
        fold_results: List of fold result dictionaries
        save_path: Path to save the plot
    """
    if not fold_results:
        print("No fold results to plot")
        return

    fold_numbers = [result['fold'] for result in fold_results]
    best_kappas = [result['best_val_kappa'] for result in fold_results]
    best_accs = [result['best_val_acc'] for result in fold_results]
    best_f1_macros = [result['best_val_f1_macro'] for result in fold_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cross-Validation Performance Summary', fontsize=16, fontweight='bold')

    metrics = [
        (best_kappas, 'Quadratic Kappa', axes[0]),
        (best_accs, 'Accuracy', axes[1]),
        (best_f1_macros, 'F1 Macro', axes[2])
    ]

    for values, label, ax in metrics:
        bars = ax.bar(fold_numbers, values, alpha=0.7, color='steelblue')

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')

        # Add std annotation
        std_val = np.std(values)
        ax.text(0.02, 0.95, f'Std: {std_val:.4f}',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8),
               fontsize=9)

        ax.set_title(f'Best Validation {label}', fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel(label)
        ax.set_xticks(fold_numbers)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Fold comparison saved to {save_path}")
    plt.show()


def plot_roc_curves(results, class_names=None, save_path=None):
    """
    Plot ROC curves for each class

    Args:
        results: Evaluation results dictionary
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    all_labels = results['true_labels']
    all_probs = results['probabilities']

    for i, class_name in enumerate(class_names):
        y_true_binary = (all_labels == i).astype(int)
        y_prob = all_probs[:, i]

        if len(np.unique(y_true_binary)) == 2:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'{class_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - One-vs-Rest', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    plt.show()

