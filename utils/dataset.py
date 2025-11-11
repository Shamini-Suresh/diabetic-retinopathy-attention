#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Dataset loading and preprocessing utilities
"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class APTOSDataset(Dataset):
    """APTOS Diabetic Retinopathy Dataset"""

    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

        # Validate CSV columns
        required_cols = ['id_code']
        if self.is_train:
            required_cols.append('diagnosis')

        for col in required_cols:
            if col not in self.labels_df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")

        if not self.is_train and 'diagnosis' not in self.labels_df.columns:
            self.labels_df['diagnosis'] = -1

        if self.is_train:
            print("Class distribution:")
            print(self.labels_df['diagnosis'].value_counts().sort_index())

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['id_code']

        # Try different extensions
        img_path = None
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

        for ext in extensions:
            potential_path = os.path.join(self.img_dir, f"{img_name}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            if torch.isnan(image).any() or torch.isinf(image).any():
                image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.is_train:
            label = self.labels_df.iloc[idx]['diagnosis']
            return image, label, img_name
        else:
            return image, -1, img_name


def prepare_datasets(train_csv, test_csv, train_img_dir, test_img_dir,
                    val_split=0.2, image_size=224):
    """Prepare train, validation, and test datasets"""

    train_df = pd.read_csv(train_csv)

    if train_df['diagnosis'].isna().any():
        train_df = train_df.dropna(subset=['diagnosis'])

    # Split train/val
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_split,
        stratify=train_df['diagnosis'],
        random_state=42
    )

    # Save temporary CSVs
    train_split_csv = 'temp_train_split.csv'
    val_csv_path = 'temp_val_split.csv'
    train_split.to_csv(train_split_csv, index=False)
    val_split.to_csv(val_csv_path, index=False)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = APTOSDataset(train_split_csv, train_img_dir, transform=train_transform, is_train=True)
    val_dataset = APTOSDataset(val_csv_path, train_img_dir, transform=test_transform, is_train=True)
    test_dataset = APTOSDataset(test_csv, test_img_dir, transform=test_transform, is_train=False)

    return train_dataset, val_dataset, test_dataset

