# ADF-Net: Asymmetric Dual-encoder Fusion Network with Physics-Color Prior

## Overview

ADF-Net is a novel deep learning network for single image dehazing, designed to effectively remove haze from images while preserving image details and colors. The network leverages asymmetric dual-encoder architecture, attention mechanisms, and physics-aware constraints to achieve high-quality dehazing results.

![ADF-Net Architecture](./picture/Figure_3.png)

## Architecture

ADF-Net consists of four main components:

1. **Asymmetric Dual-Encoder Network with Color-Structure Prior**: S-Encoder Extracts deep semantic features using residual blocks and strided convolutions，CSEM Provides clean high-frequency structural priors
2. **Physics-Guided Prediction Head (PPH)**: Estimates atmospheric light and transmission map
3. **Tri-Feature Attention Fusion Decoder (TAF)**: Integrates features from different streams using DFAB

![DFAB Architecture](./picture/Figure_4.png)


## Results

### Quantitative Performance

ADF-Net achieves state-of-the-art performance across multiple benchmark datasets:

| Dataset | PSNR | SSIM | CIEDE2000 |
|---------|------|------|-----------|
| NH-HAZE | 21.65 | 0.773 | 8.471 |
| Dense-Haze | 20.57 | 0.612 | 10.901 |
| I-HAZE | 16.79 | 0.721 | 11.466 |
| O-HAZE | 19.18 | 0.718 | 9.554 |
| StateHaze1k| 24.56 | 0.911 | 5.349 |

### Performance Scatter Slot

![Performance Scatter Slot](./picture/Figure_5.png)

### Comparison Results
- **Results on NH-HAZE**
![Results on NH-HAZE](./picture/Figure_6.png)
- **Results on Dense-Haze**
![Results on Dense-Haze](./picture/Figure_7.png)
- **Results on O-HAZE**
![Results on O-HAZE](./picture/Figure_9.png)
- **Results on StateHaze1k Thick**
![Results on O-HAZE](./picture/Figure_12.png)
## Getting Started

### Environment Setup

1. Clone this repository:

```bash
git clone 
cd ADF-Net/
```

2. Create a conda environment and install dependencies:

```bash
conda create -n adf-net python=3.8
conda activate adf-net
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Data Preparation

1. Download the following datasets:

- [StateHaze1k](https://github.com/StateHaze/StateHaze1k)

2. Organize the dataset structure as follows:

```
ADF-Net/
├── datasets_train/
│   ├── Middleburry/
│   │   ├── hazy/          # Hazy images
│   │   └── clean/         # Ground truth clear images
│   ├── I-HAZY/
│   ├── O-HAZY/
│   └── ...
├── datasets_test/
│   ├── Middleburry/
│   │   ├── hazy/          # Hazy images
│   │   └── clean/         # Ground truth clear images
│   ├── I-HAZY/
│   ├── O-HAZY/
│   └── ...
├── train_ADF.py
├── model_ADF.py
└── ...
```

## Training

### Training Command

Run the following script to train ADF-Net from scratch:

```bash
# Example: Train on Middleburry dataset
python train_ADF.py -learning_rate 1e-4 -train_batch_size 4 -train_epoch 20000 --type 3 --gpus 0
```

### Key Training Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `-learning_rate` | Learning rate | 1e-4 |
| `-train_batch_size` | Training batch size | 4 |
| `-train_epoch` | Number of training epochs | 30000 |
| `--type` | Dataset type (0-11) | 5 |
| `--gpus` | GPU device IDs | 0 |
| `--stage1_epochs` | Epochs without adversarial loss | 1000 |

### Loss Functions

ADF-Net uses a composite objective function:

| Loss Type | Weight | Description |
|-----------|--------|-------------|
| L1 Reconstruction | 1.0 | Pixel-wise reconstruction loss |
| SSIM | 0.45 | Structural similarity loss |
| Physical Consistency | 0.35 | Enforces ASM constraints |
| Contrastive Regularization | 0.1 | Enhances visual realism |
| TV Regularization | 1.0 | Transmission map smoothness |
| Identity Consistency | 0.1 | Identity mapping for clear images |

## Evaluation

### Testing Command

To evaluate the trained model on test datasets:

```bash
# Example: Test on Middleburry dataset
python test_ADF.py --type 5 --model_save_dir ./output_result_ADF_ColorHFEM/Middleburry_hazy/epoch_10000_32.5_0.95.pkl
```

### Evaluation Metrics

- **PSNR**: Measures image quality
- **SSIM**: Measures structural similarity
- **CIEDE2000**: Quantifies color differences consistent with human perception

## Citation
If you find our paper and repo are helpful for your research, please consider citing:
```bibtex

```
