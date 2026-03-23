# BBScore: Neural Benchmarking Framework

BBScore is a framework for benchmarking deep learning models against neural (fMRI, electrophysiology) and behavioral datasets. It handles model loading, stimulus preprocessing, feature extraction, and comparison with biological data.

- **Simple Notebook For Loading Data and Plotting**
[Open in Colab](https://colab.research.google.com/drive/14T5IuhZJ6PZaOvhpUMH-uIFGrzyVb7Wt?usp=sharing)
- **Data Analysis Notebook (By Josh Wilson)**
[Open in Colab](https://colab.research.google.com/drive/1Hwkro4UnmqsXWso6MEz25P1b3a0RWpbg?usp=sharing)

## Fork Contribution
 - Introduced the biologically-constrained [primate retina CNN (prCNN) model](https://github.com/Chichilnisky-Lab/primate-cnn-model);
 - Created scripts for [variance partitioning](./run_variance_partitioning.py) and [cross model analysis](./run_m2m.py);
 - Improved [run.py](./run.py) and related scripts to slice input data based on available system RAM and VRAM before loading to memory, avoiding OOM errors;
 - Performed detailed comparative analysis of prCNN versus DINOv2 in terms of primate V1 alignment and cross model alighment, details at the [project](./project) directory.

## Quick Start

### 1. Check Your System

Before installing, check if your machine can run BBScore:

```bash
python check_system.py --quick
```

For a detailed check with a specific configuration:

```bash
python check_system.py --model resnet50 --benchmark TVSDV1 --metric ridge
```

### 2. Install (Recommended: Use the Install Script)

```bash
# Make the script executable
chmod +x install.sh

# Run the interactive installer (recommended for students)
./install.sh
```

The installer features:

- **Interactive setup wizard** with arrow-key navigation
- **Tab completion** for directory paths
- **Auto-detection** of GPU (NVIDIA CUDA / Apple Silicon MPS)
- **Automatic conda/miniconda** installation if needed

**Quick install** (skip wizard, use defaults):

```bash
./install.sh --quick              # Auto-detect GPU
./install.sh --quick --cpu-only   # Force CPU-only PyTorch
```

**All options:**

```bash
./install.sh --help
```

Or install manually:

```bash
# Create conda environment
conda create -n bbscore python=3.10 -y
conda activate bbscore

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install decord from conda-forge
conda install -c conda-forge decord -y

# Install dependencies
pip install -r requirements.txt
```

### 3. Activate the Environment

If you used the install script:

```bash
# Use the generated activation script
source activate_bbscore.sh

# Or activate conda directly
conda activate bbscore
```

The install script automatically configures the `SCIKIT_LEARN_DATA` environment variable.

**Manual setup** (if not using install script):

```bash
# Required: Set data directory (50GB+ free space recommended)
export SCIKIT_LEARN_DATA="/path/to/your/data/bbscore_data"

# Add to your shell config for persistence
echo 'export SCIKIT_LEARN_DATA="/path/to/your/data/bbscore_data"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Run Your First Benchmark

```bash
# Simple example with a small model
python run.py --model resnet18 --layer _orig_mod.resnet.encoder.stages.3 --benchmark V1StaticFullFieldSineGratings --metric ridge

# Video benchmark with online metric (lower memory)
python run.py --model resnet50 --layer _orig_mod.resnet.encoder.stages.3 --benchmark OnlineTVSDV1 --metric online_linear_regressor
```

---

## For Students: Step-by-Step Guide

### **What You Need**


| Resource | Minimum          | Recommended |
| -------- | ---------------- | ----------- |
| RAM      | 8 GB             | 16+ GB      |
| GPU      | None (CPU works) | 4+ GB VRAM  |
| Disk     | 50 GB            | 100+ GB     |
| Python   | 3.9+             | 3.11        |


**No GPU?** Use:

- `ridge` metric instead of `online_linear_regressor`
- Smaller models (`resnet18`, `efficientnet_b0`)
- Online benchmarks (`OnlineTVSDV1`, `OnlineTVSDV4`)

### Validate Your Setup

Before starting your project, run the validation script to confirm your machine can handle the full pipeline. Note that this may take a while (~20 minutes) given we have to download the neural datasets.

```bash
python validate.py
```

This runs three tiers of checks:


| Tier                   | What it tests                                                                                              | Time                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **1. Environment**     | Python version, dependencies, registries, hardware (RAM, GPU, disk)                                        | ~30s                                 |
| **2. Model Inference** | Loads ResNet-18 (vision) and GPT-2 Small (language), runs forward passes                                   | ~2-3 min                             |
| **3. Data & Pipeline** | Downloads NSD and LeBel2023 datasets, validates data shapes, runs ridge and temporal RSA on synthetic data | ~5-30 min (first run downloads data) |


**All three tiers must pass before you start your project.**

You can run individual tiers to isolate issues:

```bash
python validate.py --tier 1     # Environment only
python validate.py --tier 2     # Environment + model inference
python validate.py --tier 3     # All tiers (default)
```

Expected output on a working setup:

```
  Validation Summary
  PASS  Tier 1: Environment & Dependencies  (1s)
  PASS  Tier 2: Model Loading & Inference   (45s)
  PASS  Tier 3: Data & Pipeline             (120s)

  Your machine is ready for BBScore experiments.
```

### Metric-Benchmark Compatibility

Not all metrics work with all benchmarks. The framework validates this automatically, but here is the reference:


| Benchmark Type                                 | Compatible Metrics                                                                                                                                                |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NSD, TVSD, BMD, LeBel2023** (offline neural) | `ridge`, `torch_ridge`, `pls`, `rsa`, `temporal_rsa`, `versa`, `bidirectional`, `one_to_one`, `soft_matching`, `semi_matching`, `temporal_ridge`, `inverse_ridge` |
| **LeBel2023TR** (TR-level language)            | `ridge`, `temporal_rsa`                                                                                                                                           |
| **LeBel2023Audio** (audio average)             | `ridge`, `torch_ridge`, `pls`, `rsa`, `temporal_rsa`, `versa`, `bidirectional`, `one_to_one`, `soft_matching`, `semi_matching`, `temporal_ridge`, `inverse_ridge` |
| **LeBel2023AudioTR** (TR-level audio)          | `ridge`, `temporal_rsa`                                                                                                                                           |
| **OnlineTVSD**                                 | `online_linear_regressor`                                                                                                                                         |
| **OnlinePhysionContact**                       | `physion_contact_prediction`, `physion_contact_detection`                                                                                                         |
| **OnlinePhysionPlacement**                     | `physion_placement_prediction`, `physion_placement_detection`                                                                                                     |
| **(Augmented)SSV2**                            | `online_linear_classifier`, `online_transformer_classifier`                                                                                                       |


Using an incompatible metric will print a warning with the list of compatible options.

**Note on TR-level benchmarks:** `LeBel2023TR` and `LeBel2023AudioTR` are standalone classes that bypass the standard `BenchmarkScore` pipeline. They implement their own `GroupKFold` ridge regression internally (using `sklearn.linear_model.RidgeCV` with story-level cross-validation), which is distinct from the `temporal_ridge` metric in the registry. The registry's `temporal_ridge` (`Ridge3DChunkedMetric`) expects 3D chunked features from the standard pipeline and is not compatible with TR-level benchmarks. Use `ridge` for encoding accuracy and `temporal_rsa` for representational geometry comparisons.

### Recommended Workflow

1. **Start with small experiments:**
  ```bash
   python run.py --model resnet18 --layer _orig_mod.resnet.encoder.stages.3 --benchmark NSDV1Shared --metric ridge
  ```
2. **Scale up gradually:**
  ```bash
   python run.py --model dinov2_base --layer blocks.11 --benchmark NSDV1Shared --metric ridge
  ```

---

## Available Components

### Benchmark Examples


| Benchmark                    | Type       | Memory | Description                 |
| ---------------------------- | ---------- | ------ | --------------------------- |
| `BMD_V1`                     | Video      | Low    | Macaque V1 neural responses |
| `NSDV1Shared`                | Image      | Low    | Human fMRI V1 (NSD dataset) |
| `TVSDV4`                     | Video      | High   | Macaque V4 neural responses |
| `SSV2Benchmark`              | Video      | High   | Something-Something-V2      |
| `LeBel2023{UTS01-08}`        | Text/fMRI  | Low    | Language comprehension fMRI |
| `LeBel2023TR{UTS01-08}`      | Text/fMRI  | Low    | TR-level language encoding  |
| `LeBel2023Audio{UTS01-08}`   | Audio/fMRI | Low    | Audio comprehension fMRI    |
| `LeBel2023AudioTR{UTS01-08}` | Audio/fMRI | Low    | TR-level audio encoding     |


### Models (Examples)


| Model           | Parameters | VRAM | Type  |
| --------------- | ---------- | ---- | ----- |
| `resnet18`      | 11M        | 2 GB | Image |
| `resnet50`      | 26M        | 3 GB | Image |
| `dinov2_base`   | 86M        | 4 GB | Image |
| `dinov2_large`  | 304M       | 8 GB | Image |
| `videomae_base` | 87M        | 8 GB | Video |
| `clip_vit_b32`  | 151M       | 4 GB | Image |
| `whisper_base`  | 74M        | 2 GB | Audio |
| `wav2vec2_base` | 95M        | 2 GB | Audio |
| `hubert_base`   | 95M        | 2 GB | Audio |


### Metrics


| Metric                    | GPU Required                 | Description                                 |
| ------------------------- | ---------------------------- | ------------------------------------------- |
| `ridge`                   | No                           | Ridge regression (sklearn)                  |
| `online_linear_regressor` | Recommended but not required | Online ridge with SGD and L2 regularization |
| `pls`                     | No                           | Partial Least Squares                       |
| `rsa`                     | No                           | Representational Similarity Analysis        |


---

## Command Reference

### Basic Run

```bash
python run.py --model <MODEL> --layer <LAYER> --benchmark <BENCHMARK> --metric <METRIC>
```

### Common Options

```bash
--batch-size 8       # Adjust based on your GPU memory
--device cuda:0      # Specify GPU
```

### Examples

```bash
# Image model on NSD (human fMRI)
python run.py --model resnet50 --layer _orig_mod.resnet.encoder.stages.3 --benchmark NSDV1Shared --metric ridge

# Video model on TVSD (macaque ephys)
python run.py --model videomae_base --layer encoder.layer.11 --benchmark TVSDV1 --metric ridge

# DINO on V4
python run.py --model dinov2_base --layer blocks.11 --benchmark OnlineTVSDV4 --metric ridge

# Audio model on LeBel2023 (human fMRI)
python run.py --model whisper_base --layer _orig_mod.layers.5 --benchmark LeBel2023AudioUTS01 --metric ridge

# Audio TR-level benchmark
python run.py --model wav2vec2_base --layer _orig_mod.encoder.layers.11 --benchmark LeBel2023AudioTRUTS01 --metric ridge

# Fast alpha search for high-dimensional features
python run.py --model dinov2_large --layer blocks.23 --benchmark NSDV1Shared --metric ridge --subsample-features-for-alpha 2000
```

### Finding Layer Names

To see available layer names for any model, print the model architecture:

```python
from models import MODEL_REGISTRY

# Get the model class
model_info = MODEL_REGISTRY['resnet18']
model_instance = model_info['class']()
model = model_instance.get_model('ResNet18')

# Print all layer names
for name, module in model.named_modules():
    print(name)
```

### List Available Options

```bash
python check_system.py --list
```

---

## Troubleshooting

### Out of Memory (GPU)

```bash
# Reduce batch size
python run.py ... --batch-size 2

# Use CPU
python run.py ... --device cpu --metric ridge
```

### No GPU / Installation Issues

```bash
# Reinstall with CPU-only PyTorch (smaller download, always works)
./install.sh --quick --cpu-only
```

### Out of Memory (RAM)

- Use `Online*` benchmarks instead of standard ones
- Use smaller models

### Slow Training

- Install cuML for 50x faster ridge regression: [https://docs.rapids.ai/api/cuml/stable/](https://docs.rapids.ai/api/cuml/stable/)
- Use `--subsample-features-for-alpha 2000` for faster alpha search

### Dataset Download Issues

- Ensure `SCIKIT_LEARN_DATA` is set to a writable directory
- Check you have enough disk space
- Some datasets require AWS credentials (see `data/` folder)

---

## Loss Functions for OnlineLinearRegressor

When using `OnlineLinearRegressor`, you can choose different loss functions:


| Loss Type     | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `mse`         | **Default** - Mean squared error with L2 regularization            |
| `correlation` | Pearson correlation loss                                           |
| `combined`    | MSE + correlation (tune `correlation_weight`)                      |
| `ccc`         | Concordance Correlation Coefficient (combines correlation + scale) |
| `ccc_mse`     | CCC + MSE combined                                                 |


**Example:**

```python
from metrics import OnlineLinearRegressor

# Default configuration (MSE + L2)
metric = OnlineLinearRegressor(
    input_feature_dim=768,
    loss_type='mse',  # Default
    n_epochs=100,
)

# Alternative: CCC loss is also available
metric_ccc = OnlineLinearRegressor(
    input_feature_dim=768,
    loss_type='ccc',
    n_epochs=100,
)
```

---

## Project Structure

```
bbscore_public/
├── benchmarks/          # Benchmark definitions (NSD, TVSD, Physion, etc.)
├── data/                # Dataset loaders and downloaders
├── metrics/             # Scoring methods (ridge, RSA, PLS, online)
│   └── losses.py        # Loss functions (MSE, CCC, Pearson, etc.)
├── models/              # Model wrappers (HuggingFace, TorchVision)
├── run.py               # Main entry point
├── eval.py              # Batch evaluation script
├── check_system.py      # System diagnostic tool
├── install.sh           # Interactive installation script
├── activate_bbscore.sh  # Environment activation (generated by install.sh)
└── requirements.txt     # Python dependencies
```

---

## Getting Help

1. **Check your system:** `python check_system.py`
2. **List options:** `python check_system.py --list`

