#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Training script for vanilla (baseline) models without attention
Usage: python train_vanilla.py --backbone resnet50, densenet121,efficientnetb0 --epochs 25
"""

from utils.seed import set_all_seeds, set_worker_seeds
set_all_seeds(42)

import argparse
import os
import torch
from torch.utils.data import DataLoader

from models import APTOSVanillaNet
from utils import prepare_datasets, train_kfold, evaluate_model, plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description='Train vanilla DR classification model')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                       choices=['resnet50', 'densenet121', 'efficientnet_b0'],
                       help='Backbone architecture')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--train_images', type=str, required=True, help='Path to train images')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--test_images', type=str, required=True, help='Path to test images')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for CV')
    parser.add_argument('--save_dir', type=str, default='./results_vanilla', help='Save directory')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training Vanilla {args.backbone} on {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        args.train_csv, args.test_csv,
        args.train_images, args.test_images
    )

    # Train model
    best_model, fold_histories, fold_results = train_kfold(
        model_class=APTOSVanillaNet,
        backbone=args.backbone,
        device=device,
        train_dataset=train_dataset,
        k_folds=args.k_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_attention=False
    )

    # Evaluate
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    val_results = evaluate_model(best_model, val_loader, device)

    # Visualize
    plot_training_history(fold_histories, save_path=os.path.join(args.save_dir, 'history.png'))

    # Save model
    torch.save(best_model.state_dict(), os.path.join(args.save_dir, f'best_model_{args.backbone}.pth'))

    print(f"\n✅ Training complete! Results saved to {args.save_dir}")
    print(f"Quadratic Kappa: {val_results['quadratic_kappa']:.4f}")


if __name__ == '__main__':
    main()

