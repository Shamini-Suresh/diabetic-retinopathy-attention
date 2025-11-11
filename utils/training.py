#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Training utilities for k-fold cross-validation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, f1_score
from tqdm import tqdm
import contextlib


def train_kfold(model_class, backbone, device, train_dataset, k_folds=5, epochs=25, 
                batch_size=16, use_attention=False, attention_reduction=16):
    """
    Train model with k-fold cross-validation

    Args:
        model_class: Model class (APTOSVanillaNet or APTOSAttentionNet)
        backbone: Backbone architecture name
        device: torch device
        train_dataset: Training dataset
        k_folds: Number of folds for CV
        epochs: Number of training epochs
        batch_size: Batch size
        use_attention: Whether to use attention (for APTOSAttentionNet)
        attention_reduction: Attention reduction ratio

    Returns:
        best_model, fold_histories, fold_results
    """
    # Get labels for stratified split
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")

    # Storage for results
    fold_histories = []
    fold_results = []
    best_val_kappa = 0
    best_model_state = None

    for fold, (train_ids, val_ids) in enumerate(skfold.split(np.zeros(len(labels)), labels)):
        print(f'\n{"="*60}')
        print(f'FOLD {fold+1}/{k_folds}')
        print(f'Training samples: {len(train_ids)}, Validation samples: {len(val_ids)}')
        print(f'{"="*60}')

        # Prepare data loaders
        train_labels = [labels[i] for i in train_ids]
        train_weights = [class_weights[label] for label in train_labels]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        train_loader = DataLoader(
            Subset(train_dataset, train_ids),
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=device.type == 'cuda',
            persistent_workers=True,
            drop_last=True
        )

        val_loader = DataLoader(
            Subset(train_dataset, val_ids),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=device.type == 'cuda',
            persistent_workers=True,
            drop_last=False
        )

        # Create model
        if use_attention:
            model = model_class(
                backbone=backbone,
                dropout_rate=0.4,
                num_classes=5,
                use_attention=True,
                attention_reduction=attention_reduction
            ).to(device)
        else:
            model = model_class(
                backbone=backbone,
                dropout_rate=0.4,
                num_classes=5
            ).to(device)

        # Loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # Early stopping
        patience, counter = 15, 0
        fold_best_kappa = 0

        # Fold history
        fold_history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'val_kappa': [], 'val_weighted_kappa': [], 'val_f1_macro': [], 'val_f1_weighted': []
        }

        # Training loop
        for epoch in range(epochs):
            # Train phase
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for batch_idx, (images, labels_batch, _) in enumerate(tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{epochs}")):
                images, labels_batch = images.to(device), labels_batch.to(device)

                if torch.isnan(images).any() or torch.isinf(images).any():
                    continue

                optimizer.zero_grad()

                if scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels_batch)

                        if torch.isnan(loss) or torch.isinf(loss):
                            continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels_batch).sum().item()
                train_total += labels_batch.size(0)

            # Validation phase
            model.eval()
            val_loss, val_correct, val_total, val_preds, val_labels = 0, 0, 0, [], []

            with torch.no_grad():
                for images, labels_batch, _ in tqdm(val_loader, desc="Validating"):
                    images, labels_batch = images.to(device), labels_batch.to(device)

                    if torch.isnan(images).any() or torch.isinf(images).any():
                        continue

                    with torch.amp.autocast(device_type='cuda') if scaler else contextlib.nullcontext():
                        outputs = model(images)
                        loss = criterion(outputs, labels_batch)

                        if torch.isnan(loss) or torch.isinf(loss):
                            continue

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels_batch).sum().item()
                    val_total += labels_batch.size(0)
                    val_preds.extend(outputs.argmax(1).cpu().numpy())
                    val_labels.extend(labels_batch.cpu().numpy())

            scheduler.step()

            # Calculate metrics
            avg_train_loss = train_loss / max(len(train_loader), 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)
            train_accuracy = train_correct / max(train_total, 1)
            val_accuracy = val_correct / max(val_total, 1)

            if len(val_preds) > 0 and len(val_labels) > 0:
                val_kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
                val_weighted_kappa = cohen_kappa_score(val_labels, val_preds, weights='linear')
                val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
                val_f1_weighted = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            else:
                val_kappa = val_weighted_kappa = val_f1_macro = val_f1_weighted = 0.0

            # Store metrics
            fold_history['train_loss'].append(avg_train_loss)
            fold_history['train_acc'].append(train_accuracy)
            fold_history['val_loss'].append(avg_val_loss)
            fold_history['val_acc'].append(val_accuracy)
            fold_history['val_kappa'].append(val_kappa)
            fold_history['val_weighted_kappa'].append(val_weighted_kappa)
            fold_history['val_f1_macro'].append(val_f1_macro)
            fold_history['val_f1_weighted'].append(val_f1_weighted)

            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            print(f'  Val Kappa: {val_kappa:.4f}, F1 Macro: {val_f1_macro:.4f}')

            # Update best
            if val_kappa > fold_best_kappa:
                fold_best_kappa = val_kappa

            if val_kappa > best_val_kappa and not (avg_val_loss > avg_train_loss * 3.0) and not np.isnan(val_kappa):
                best_val_kappa = val_kappa
                best_model_state = model.state_dict().copy()
                counter = 0
                print(f'✓ New best validation kappa: {val_kappa:.4f}')
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Store fold results
        fold_histories.append(fold_history)
        fold_results.append({
            'fold': fold + 1,
            'best_val_kappa': fold_best_kappa,
            'best_val_acc': max(fold_history['val_acc']) if fold_history['val_acc'] else 0,
            'best_val_f1_macro': max(fold_history['val_f1_macro']) if fold_history['val_f1_macro'] else 0,
            'history': fold_history
        })

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Create best model
    if use_attention:
        best_model = model_class(
            backbone=backbone,
            dropout_rate=0.4,
            num_classes=5,
            use_attention=True,
            attention_reduction=attention_reduction
        ).to(device)
    else:
        best_model = model_class(
            backbone=backbone,
            dropout_rate=0.4,
            num_classes=5
        ).to(device)

    if best_model_state:
        best_model.load_state_dict(best_model_state)

    print(f"\n🏆 Best validation kappa: {best_val_kappa:.4f}")

    return best_model, fold_histories, fold_results

