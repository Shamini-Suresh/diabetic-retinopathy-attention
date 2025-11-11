#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Model evaluation utilities with comprehensive metrics
"""
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report
)
import contextlib


def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """
    Calculate confidence interval for a metric using bootstrap sampling

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Metric function to evaluate
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        lower, upper: Confidence interval bounds
    """
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]

        try:
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue

    if len(bootstrap_scores) == 0:
        return np.nan, np.nan

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)

    return lower, upper


def evaluate_model(model, data_loader, device, class_names=None, confidence_level=0.95):
    """
    Comprehensive model evaluation with per-class metrics and confidence intervals

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: torch device
        class_names: List of class names
        confidence_level: Confidence level for intervals

    Returns:
        results: Dictionary containing all evaluation metrics
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(data_loader, desc="Evaluating"):
            # Skip samples without labels
            if (labels == -1).all():
                continue

            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else contextlib.nullcontext():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if len(all_preds) == 0:
        print("No labeled samples found for evaluation")
        return {}

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate basic metrics
    accuracy = accuracy_score(all_labels, all_preds)
    quadratic_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    linear_kappa = cohen_kappa_score(all_labels, all_preds, weights='linear')
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

    # Per-class AUC scores
    auc_scores = []
    per_class_ap = []

    for i in range(len(class_names)):
        try:
            y_true_binary = (all_labels == i).astype(int)
            y_prob = all_probs[:, i]

            if len(np.unique(y_true_binary)) == 2:
                auc = roc_auc_score(y_true_binary, y_prob)
                ap = average_precision_score(y_true_binary, y_prob)
                auc_scores.append(auc)
                per_class_ap.append(ap)
            else:
                auc_scores.append(np.nan)
                per_class_ap.append(np.nan)
        except Exception as e:
            auc_scores.append(np.nan)
            per_class_ap.append(np.nan)

    mean_auc = np.nanmean(auc_scores)
    mean_ap = np.nanmean(per_class_ap)

    # Calculate confidence intervals
    print("Calculating confidence intervals...")

    def accuracy_func(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def kappa_func(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    def f1_macro_func(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

    acc_ci_lower, acc_ci_upper = bootstrap_confidence_interval(
        all_labels, all_preds, accuracy_func, confidence=confidence_level)

    kappa_ci_lower, kappa_ci_upper = bootstrap_confidence_interval(
        all_labels, all_preds, kappa_func, confidence=confidence_level)

    f1_ci_lower, f1_ci_upper = bootstrap_confidence_interval(
        all_labels, all_preds, f1_macro_func, confidence=confidence_level)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names,
                                         zero_division=0, output_dict=True)

    # Per-class support
    per_class_support = [np.sum(all_labels == i) for i in range(len(class_names))]

    results = {
        # Basic metrics
        'accuracy': accuracy,
        'quadratic_kappa': quadratic_kappa,
        'linear_kappa': linear_kappa,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mean_auc': mean_auc,
        'mean_ap': mean_ap,

        # Per-class metrics
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_auc': auc_scores,
        'per_class_ap': per_class_ap,
        'per_class_support': per_class_support,

        # Confidence intervals
        'accuracy_ci': (acc_ci_lower, acc_ci_upper),
        'kappa_ci': (kappa_ci_lower, kappa_ci_upper),
        'f1_macro_ci': (f1_ci_lower, f1_ci_upper),

        # Other
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }

    # Print results
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS WITH CONFIDENCE INTERVALS")
    print(f"{'='*80}")
    print(f"Overall Metrics:")
    print(f"  Accuracy: {accuracy:.4f} [{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]")
    print(f"  Quadratic Kappa: {quadratic_kappa:.4f} [{kappa_ci_lower:.4f}, {kappa_ci_upper:.4f}]")
    print(f"  Linear Kappa: {linear_kappa:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f} [{f1_ci_lower:.4f}, {f1_ci_upper:.4f}]")
    print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"  Mean AUC: {mean_auc:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Support':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print(f"{'-'*70}")

    for i, class_name in enumerate(class_names):
        support = per_class_support[i]
        f1 = per_class_f1[i] if i < len(per_class_f1) else 0.0
        precision = per_class_precision[i] if i < len(per_class_precision) else 0.0
        recall = per_class_recall[i] if i < len(per_class_recall) else 0.0

        print(f"{class_name:<20} {support:<10} {f1:<10.4f} {precision:<12.4f} {recall:<10.4f}")

    return results

