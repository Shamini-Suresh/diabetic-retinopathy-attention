
# Attention Mechanisms in CNNs for Diabetic Retinopathy Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of "Architecture-Dependent Effects of Attention Mechanisms in Deep Learning for Diabetic Retinopathy Grading"

Summary: We discovered that attention mechanisms don't always improve performance. They significantly degrade ResNet-50 and DenseNet-121 performance but work well with EfficientNet-B0 for diabetic retinopathy classification.

Key Findings

Our research reveals architecture-dependent effects of attention mechanisms on diabetic retinopathy classification:

| Architecture | Baseline κ | With Attention | Change | 95% CI | Significance |
|--------------|-----------|----------------|--------|--------|--------------|
| ResNet-50 | 0.8682 | 0.8420 | -3.0% | [0.837, 0.892] vs [0.812, 0.868] | p < 0.05 |
| DenseNet-121| 0.8397 | 0.8080 | -3.8% | 0.811, 0.867] vs. [0.777, 0.836] | p < 0.05 |
| EfficientNet-B0 | 0.8831 | 0.8890 | +0.7% | [0.855, 0.910] vs. [0.861, 0.912] | not significant |

Main Contributions

1. First systematic study of attention mechanism effects across multiple CNN architectures for medical imaging. 
2. Identified architecture-specific incompatibilities: Attention disrupts ResNet’s residual learning and DenseNet’s dense connections.  
3. Clinical implications: Attention degrades moderate DR detection (F1: 0.68 → 0.50), the most critical class for intervention.  
4. Actionable insights: EfficientNet-B0’s compound scaling is naturally compatible with attention mechanisms.  

Quick Start
Installation

```bash
git clone [https://github.com/Shamini-Suresh/diabetic-retinopathy-attention.git](https://github.com/Shamini-Suresh/diabetic-retinopathy-attention.git)
cd diabetic-retinopathy-attention
pip install -r requirements.txt
````
Dataset

Download the [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data) from Kaggle. The expected data structure is:

```
data/
├── train.csv
├── test.csv
├── train_images/
└── test_images/
```

Train Baseline Model

```bash
python train_vanilla.py \
    --backbone efficientnet_b0 \
    --train_csv data/train.csv \
    --train_images data/train_images \
    --test_csv data/test.csv \
    --test_images data/test_images \
    --epochs 25 \
    --batch_size 16 \
    --save_dir results/vanilla/efficientnet_b0
```

Train Attention Model

```bash
python train_attention.py \
    --backbone efficientnet_b0 \
    --train_csv data/train.csv \
    --train_images data/train_images \
    --test_csv data/test.csv \
    --test_images data/test_images \
    --epochs 25 \
    --batch_size 16 \
    --attention_reduction 16 \
    --save_dir results/attention/efficientnet_b0
```
Repository Structure

```
├── models/                      
│   ├── __init__.py
│   ├── vanilla_model.py        
│   └── attention_model.py      
│
├── utils/                       
│   ├── __init__.py
│   ├── dataset.py              
│   ├── training.py             
│   ├── evaluation.py           
│   └── visualization.py        
│
├── train_vanilla.py            
├── train_attention.py          
├── requirements.txt            
├── LICENSE                     
└── README.md                   
```

Methodology
Model Architectures

  * Backbones: ResNet-50, DenseNet-121, EfficientNet-B0
  * Attention Mechanism: Dual attention module combining:
      * Spatial Attention: Learns *where* to focus in the image
      * Channel Attention: Learns *which* features are important
  * Classifier: 3-layer MLP with dropout (rates: 0.4, 0.2, 0.1)

Training Configuration

| Parameter | Value |
| :--- | :--- |
| Dataset | APTOS 2019 (3,662 training images) |
| Classes | 5 (No DR, Mild, Moderate, Severe, Proliferative) |
| Cross-Validation | 5-fold stratified |
| Optimizer | AdamW (lr=5e-4, weight\_decay=1e-4) |
| Scheduler | Cosine Annealing Warm Restarts |
| Image Size | 224×224 |
| Batch Size | 16 |
| Epochs | 25 (early stopping, patience=15) |
| *Augmentation | Random flip, rotation (±10°), color jitter |
| Class Weighting | Balanced (inverse frequency) |
| Label Smoothing | 0.05 |

Evaluation Metrics

  * Primary: Quadratic Weighted Kappa (κ)
  * Secondary: Accuracy, F1 (macro/weighted), per-class precision/recall
  * Statistical Validation: Bootstrap confidence intervals (1,000 iterations, 95% CI)
  * Visualization: Confusion matrices, ROC curves, attention maps

Reproduce Paper Results

```bash
# Train all baseline models
for backbone in resnet50 densenet121 efficientnet_b0; do
    python train_vanilla.py \
        --backbone $backbone \
        --train_csv data/train.csv \
        --train_images data/train_images \
        --test_csv data/test.csv \
        --test_images data/test_images \
        --epochs 25 \
        --save_dir results/vanilla/$backbone
done

# Train all attention models
for backbone in resnet50 densenet121 efficientnet_b0; do
    python train_attention.py \
        --backbone $backbone \
        --train_csv data/train.csv \
        --train_images data/train_images \
        --test_csv data/test.csv \
        --test_images data/test_images \
        --epochs 25 \
        --save_dir results/attention/$backbone
done
```

License

This project is licensed under the MIT License - see the `LICENSE` file for details.

Acknowledgments

  * [APTOS 2019 Blindness Detection Challenge](https://www.kaggle.com/c/aptos2019-blindness-detection) organizers and participants
  * [PyTorch](https://pytorch.org/) team for the deep learning framework
  * `torchvision` for pre-trained model weights
  * [Kaggle](https://www.kaggle.com/) for hosting the dataset
  * All contributors and reviewers.
