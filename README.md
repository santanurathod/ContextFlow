# ContextFlow Training and Evaluation Pipeline

This repository contains code and instructions for processing spatiotemporal omics datasets and running experiments with ContextFlow framework.

The guide explains how to train and evaluate trajectory inference models using ContextFlow.

## Prerequisites

Complete Steps 1-2 from the datasets README to download and preprocess data.

## Pipeline Overview

### 1. Download and Preprocess Datasets

Follow the instructions in the `datasets/` folder README to:
- Download raw datasets
- Preprocess and generate `.h5ad` files

### 2. Compute Celltype Encoder (for Weighted Wasserstein)

Generate the celltype encoder required for computing weighted Wasserstein distances:

```bash
python src/create_celltype_encoder.py --h5ad_path <file_from_step1.h5ad>
```

**Example:**
```bash
python src/create_celltype_encoder.py --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad
```

> `<file_from_step1.h5ad>` refers to the preprocessed `.h5ad` file created in Step 1 (e.g., `GSE232025_stereoseq_g_10000_nzp_0.1.h5ad`)

### 3. Train and Evaluate Models

#### Model Variants

- **CTF-H**: ContextFlow with entropic regularization
- **MOTFM**: Minibatch-OT Flow Matching (baseline)

#### Parameters

- `--train_config`: Model configuration specifying OT coupling type and hyperparameters (see `train_configs/README.md` for details)
- `--h5ad_path`: Path to the preprocessed dataset file
- `--new_experiment`: Experiment/folder name for saving validation metrics and trained models
- `--interpolation`: Set to `True` for interpolation tasks
- `--train_idx`: Indices of time points for training
- `--test_idx`: Index of held-out time point for testing

---

## Training Commands

### CTF-H (λ = 1)

#### Interpolation
```bash
python main.py \
    --train_config vi_pca_C_g_REOT1_mc \
    --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad \
    --new_experiment v_interp_post_prior_correction_REOT_w_C_g_reg_mc_ti2 \
    --interpolation True \
    --train_idx 0 1 3 4 \
    --test_idx 2
```

#### Extrapolation
```bash
python main.py \
    --train_config vi_pca_C_g_REOT1_mc \
    --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad \
    --new_experiment v_post_prior_correction_REOT_w_C_g_reg_mc
```

---

### CTF-H (λ = 0)

#### Interpolation
```bash
python main.py \
    --train_config vi_pca_C_g_REOT1_lr \
    --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad \
    --new_experiment v_interp_post_prior_correction_REOT_w_C_g_reg_lr_ti2 \
    --interpolation True \
    --train_idx 0 1 3 4 \
    --test_idx 2
```

#### Extrapolation
```bash
python main.py \
    --train_config vi_pca_C_g_REOT1_lr \
    --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad \
    --new_experiment v_post_prior_correction_REOT_w_C_g_reg_lr
```

---

### MOTFM (Baseline)

#### Interpolation
```bash
python main.py \
    --train_config vi_pca_C_g_EOT \
    --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad \
    --new_experiment v_interp_post_prior_correction_EOT_w_g_ti2 \
    --interpolation True \
    --train_idx 0 1 3 4 \
    --test_idx 2
```

#### Extrapolation
```bash
python main.py \
    --train_config vi_pca_C_g_EOT \
    --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad \
    --new_experiment v_post_prior_correction_EOT_w_g
```

---

## Configuration Details

For detailed information about training configurations and hyperparameters, refer to `train_configs/README.md`.

## Output

Trained models and validation metrics are saved in directories named according to the `--new_experiment` parameter.

## License

[MIT License](LICENSE)