# RailVisionX

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**RailVisionX** is an advanced computer vision pipeline for real-time and offline video processing, featuring state-of-the-art image deblurring using a MobileNetV2-based architecture with adversarial training (DeblurGAN-v2). The system leverages the lightweight MobileNetV2 backbone combined with Feature Pyramid Networks for efficient multi-scale feature extraction, optimized for edge deployment via TensorRT.

---

## Key Features

- ** MobileNetV2 Architecture**: Lightweight and efficient deblurring model using MobileNetV2 backbone with Feature Pyramid Network and adversarial training
- ** Real-time Processing**: TensorRT-optimized inference for edge deployment with sub-20ms latency
- ** Analytics Dashboard**: Comprehensive monitoring and visualization of image quality metrics
- ** Multi-task Pipeline**: Seamlessly integrates object detection, OCR, and tracking on enhanced images
- ** Production-Ready**: Complete training, evaluation, and deployment pipeline with extensive documentation

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Evaluation & Metrics](#evaluation--metrics)
- [Performance](#performance)
- [License](#license)

---

##  Architecture

### MobileNetV2-Based Deblurring Network

The system uses **MobileNetV2** as the core backbone architecture with adversarial training for enhanced performance:

```
Input (Blurred Image)
         ↓
    Deblurring Network
    ├── MobileNetV2 Backbone (Pre-trained encoder)
    ├── Feature Pyramid Network (Multi-scale fusion)
    └── Decoder (Transposed convolutions)
         ↓
    Output (Sharp Image)
```

**Training Methodology:**
The model is trained with adversarial learning using a discriminator network:
```
Adversarial Training Loop:
  1. MobileNetV2 Network generates deblurred image
  2. Discriminator evaluates image quality
  3. Combined loss optimizes both networks
```

**Loss Function:**
```
Total Loss = λ_L1 × L1_Loss + λ_Perceptual × VGG_Loss + λ_Adversarial × Adversarial_Loss
```

**Key Components:**
- **MobileNetV2 Backbone**: Efficient feature extraction with depthwise separable convolutions
- **Feature Pyramid Network (FPN)**: Multi-scale feature fusion for handling various blur levels
- **Discriminator Network**: PatchGAN architecture used during training for adversarial learning
- **Perceptual Loss**: VGG19-based feature matching for improved visual quality

**Why MobileNetV2?**
- **Lightweight**: Only ~3.5M parameters in the encoder
- **Fast**: Optimized for mobile and edge devices
- **Efficient**: Depthwise separable convolutions reduce computation
- **Proven**: Pre-trained on ImageNet provides excellent feature extraction

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory recommended (16GB for optimal training)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/RailVisionX.git
   cd RailVisionX
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

---

## Quick Start

### 1. Prepare Your Data

Organize your image pairs with matching filenames:

```
data/raw/
├── blurred/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── sharp/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

### 2. Create Data Splits

Split your dataset into training (70%), validation (10%), and test (20%) sets:

```bash
python src/dataset/make_splits.py
```

This creates:
```
data/processed/
├── train/
├── val/
└── test/
```

### 3. Train the Model

**Basic training:**
```bash
python apps/train_deblur.py --batch_size 8 --num_epochs 100
```

**Advanced training with custom configuration:**
```bash
python apps/train_deblur.py \
    --data_dir data/processed \
    --output_dir outputs/deblur_training \
    --image_size 256 \
    --batch_size 16 \
    --num_epochs 150 \
    --lr_g 1e-4 \
    --lr_d 1e-4 \
    --lambda_l1 100.0 \
    --lambda_perceptual 1.0 \
    --device cuda
```

### 4. Monitor Training

Launch TensorBoard to visualize training progress:

```bash
tensorboard --logdir outputs/deblur_training/logs
```

Access the dashboard at: http://localhost:6006

### 5. Evaluate Performance

Run evaluation on the test set:

```bash
python apps/evaluate_deblur.py --checkpoint outputs/deblur_training/best_model.pth
```

### 6. Export for Deployment

Convert the trained model to ONNX format:

```bash
python apps/export_onnx.py \
    --checkpoint outputs/deblur_training/best_model.pth \
    --output models/deblur_generator.onnx
```

---

## Project Structure

```
RailVisionX/
├── apps/                           # Application entry points
│   ├── train_deblur.py            # Model training script
│   ├── evaluate_deblur.py         # Model evaluation
│   ├── export_onnx.py             # ONNX export
│   ├── edge_realtime.py           # Real-time inference
│   ├── offline_process.py         # Batch processing
│   └── dashboard.py               # Analytics dashboard
│
├── src/                           # Core library code
│   ├── dataset/                   # Data processing
│   │   ├── make_splits.py        # Dataset splitting
│   │   ├── dataloader.py         # PyTorch data loaders
│   │   └── augment.py            # Data augmentation
│   │
│   ├── enhancement/               # Image enhancement models
│   │   ├── deblur_net.py         # MobileNetV2 deblurring architecture
│   │   ├── denoise.py            # Noise reduction
│   │   └── lowlight_enhance.py   # Low-light enhancement
│   │
│   ├── inference/                 # Inference engines
│   │   ├── trt_runner.py         # TensorRT inference
│   │   └── deepstream/           # DeepStream integration
│   │
│   ├── quality/                   # Quality assessment
│   │   ├── blur_score.py         # Blur detection
│   │   └── lowlight_score.py     # Lighting analysis
│   │
│   └── common/                    # Shared utilities
│       ├── config.py             # Configuration management
│       ├── logger.py             # Logging utilities
│       └── utils.py              # Helper functions
│
├── configs/                       # Configuration files
│   ├── app.yaml                  # Application settings
│   └── cameras.yaml              # Camera configurations
│
├── data/                         # Dataset storage
│   ├── raw/                     # Original data
│   └── processed/               # Preprocessed splits
│
├── models/                       # Trained models
│   ├── *.pth                    # PyTorch checkpoints
│   ├── *.onnx                   # ONNX exports
│   └── *.trt                    # TensorRT engines
│
├── outputs/                      # Training outputs
│   ├── deblur_training/         # Training artifacts
│   │   ├── logs/               # TensorBoard logs
│   │   └── *.pth               # Checkpoints
│   └── evaluation/              # Evaluation results
│
├── scripts/                      # Helper scripts
│   ├── run_preprocess.sh        # Data preprocessing
│   ├── run_train.sh             # Training launcher
│   └── run_export_onnx.sh       # Model export
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Training Pipeline

### Data Preparation

The training pipeline expects paired blurred/sharp images:

1. **Collect Data**: Gather pairs of blurred inputs and sharp ground truth images
2. **Naming Convention**: Ensure matching filenames (e.g., `img_001.jpg` in both folders)
3. **Split Dataset**: Run `make_splits.py` to create train/val/test splits
4. **Augmentation**: Automatic augmentation applied during training (flips, rotations, crops)

### Training Configuration

Key hyperparameters (adjustable via command line or `configs/app.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 256 | Input/output image resolution |
| `batch_size` | 8 | Training batch size |
| `num_epochs` | 100 | Number of training epochs |
| `lr_g` | 1e-4 | MobileNetV2 network learning rate |
| `lr_d` | 1e-4 | Discriminator learning rate (training only) |
| `lambda_l1` | 100.0 | L1 reconstruction loss weight |
| `lambda_perceptual` | 1.0 | Perceptual loss weight |
| `lambda_adv` | 1.0 | Adversarial loss weight |

### Training Workflow

```python
# 1. Initialize trainer
trainer = DeblurGANTrainer(
    data_dir='data/processed',
    output_dir='outputs/training',
    image_size=256,
    batch_size=8
)

# 2. Train model
trainer.train(num_epochs=100, save_every=10)

# 3. Best model saved automatically based on validation loss
# Located at: outputs/training/best_model.pth
```

---

## Evaluation & Metrics

### Supported Metrics

The evaluation pipeline computes comprehensive quality metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Pixel-level reconstruction accuracy
  - Excellent: > 30 dB
  - Good: 25-30 dB
  - Fair: 20-25 dB
  
- **SSIM (Structural Similarity Index)**: Structural similarity
  - Range: 0-1 (higher is better)
  - Target: > 0.85 for good quality

- **MAE (Mean Absolute Error)**: Average pixel difference
  - Lower is better
  
- **Inference Time**: Processing latency (ms per image)

### Running Evaluation

```bash
python apps/evaluate_deblur.py \
    --checkpoint outputs/deblur_training/best_model.pth \
    --data_dir data/processed \
    --output_dir outputs/evaluation
```

**Output:**
- Quantitative metrics (PSNR, SSIM, MAE)
- Visual comparisons (blurred/deblurred/sharp)
- Metrics CSV file for detailed analysis
- Distribution plots and correlation graphs

---

## Resuming Training

```python
import torch
from apps.train_deblur import DeblurGANTrainer

trainer = DeblurGANTrainer(...)

# Load checkpoint
checkpoint = torch.load('outputs/training/checkpoint_epoch_50.pth')
trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

# Continue training
trainer.train(num_epochs=50)
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
