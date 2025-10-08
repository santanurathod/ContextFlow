# Training Configuration Guide

This document explains the parameters in the training configuration files located in `train_configs/post_prior_correction/`.

## Example Configuration

```json
{
    "use_celltype_conditional": false,
    "use_spatial": false,
    "use_bio_prior": false,
    "OT_cost_variation": "g",
    "OT_reg_variation": "relativeentropic_g+mc+lr",
    "representation": "pca",
    "g_normalize": false,
    "lr_normalize": true,
    "mc_normalize": true,
    "ot_method": "sinkhorn_relative_entropy",
    "cc_communication_type": "step_by_step",
    "entropy_reg": 100,
    "OT_reg_lambda": 0.4,
    "cfm_model": "MLP",
    "lambda_": 0,
    "lambda_bio_prior": 0,
    "batch_size": 256,
    "sigma": 0.9,
    "learning_rate": 1e-4,
    "n_epochs": 10000,
    "w": 10,
    "time_varying": true,
    "trajectory_steps": 400
}
```

## Parameter Descriptions

### Model Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `use_celltype_conditional` | bool | Whether to use celltype labels as conditional inputs to the velocity field |
| `use_spatial` | bool | Whether to use actual spatial coordinates in the prior matrix. **Default: `false`** as coordinate frames are not aligned across time points |
| `use_bio_prior` | bool | Whether to use celltype-level feasibility hard prior. **Default: `false`** |

### OT Configuration

#### Cost Matrix (`OT_cost_variation`)

Specifies what data to use in computing the OT cost matrix:

- **`"g"`**: Only transcriptomic information
- **`"g+mc"`**: Transcriptomic + spatial smoothness prior
- **`"g+mc+lr"`**: Transcriptomic + spatial smoothness prior + ligand-receptor communication patterns

#### Regularization (`OT_reg_variation`)

Specifies the type of entropic regularization:

- **`"entropic"`**: Default regularization used in standard Entropic-OT formulations
- **`"relativeentropic_g+mc+lr"`**: ContextFlow regularization using spatial smoothness (mc) + ligand-receptor patterns (lr)
- **`"relativeentropic_g+mc"`**: ContextFlow regularization using only spatial smoothness (mc)
- **`"relativeentropic_g+lr"`**: ContextFlow regularization using only ligand-receptor patterns (lr)

#### OT Algorithm (`ot_method`)

- **`"sinkhorn"`**: Default Sinkhorn algorithm
- **`"sinkhorn_relative_entropy"`**: Modified Sinkhorn algorithm for ContextFlow

### Feature Representation

| Parameter | Type | Description |
|-----------|------|-------------|
| `representation` | str | Feature encoding method (e.g., `"pca"`) |
| `g_normalize` | bool | Normalize transcriptomic feature vector by its maximum |
| `lr_normalize` | bool | Normalize ligand-receptor communication feature vector by its maximum |
| `mc_normalize` | bool | Normalize spatial smoothness feature vector by its maximum |

### Hyperparameters

#### ContextFlow Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `entropy_reg` | float | Regularization parameter (ε) for entropy term in OT formulation |
| `OT_reg_lambda` | float | **λ** from ContextFlow paper: tunes spatial smoothness (MC) vs ligand-receptor (LR) in transition plausibility matrix |
| `lambda_` | float | **α** from ContextFlow paper: tunes transcriptomic cost vs biological prior<br>Prior = `λ × mc + (1-λ) × LR` |
| `lambda_bio_prior` | float | Weight for biological prior term |

#### Neural Network Architecture

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfm_model` | str | Neural network architecture for velocity field `v_θ` (e.g., `"MLP"`) |
| `w` | int | Width of the CFM model |

#### Flow Matching Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sigma` | float | Variance of the probability path for flow matching framework |
| `time_varying` | bool | Whether to use time-varying velocity field |
| `trajectory_steps` | int | Number of discretization steps for sampling during validation/inference |

#### Training Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch_size` | int | Training batch size |
| `learning_rate` | float | Optimization learning rate |
| `n_epochs` | int | Number of training epochs |

### Other Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cc_communication_type` | str | Cell-cell communication computation method (e.g., `"step_by_step"`) |

## Configuration Naming Convention

Configuration files follow the pattern: `vi_pca_C_{cost}_{reg}_{normalization}.json`

**Example:** `vi_pca_C_g_REOT1_mc+lr.json`
- `vi`: Version identifier
- `pca`: Feature representation
- `C`: Refers to cost, followed by the kind of cost
- `g`: Cost variation (transcriptomic only); g+mc/g+lr/g+mc+lr refer to appropriate variations
- `REOT1`: Relative entropic OT with specific hyperparameters
- `mc+lr`: Normalization includes spatial smoothness + ligand-receptor patterns

## Quick Reference: Key Model Variants

| Model | Config Pattern | Cost | Regularization | Description |
|-------|---------------|------|----------------|-------------|
| **CTF-H (λ=1)** | `vi_pca_C_g_REOT1_mc` | `g` | `relativeentropic_g+mc` | ContextFlow with spatial smoothness only |
| **CTF-H (λ=0)** | `vi_pca_C_g_REOT1_lr` | `g` | `relativeentropic_g+lr` | ContextFlow with LR patterns only |
| **MOTFM** | `vi_pca_C_g_EOT` | `g` | `entropic` | Baseline minibatch-OT flow matching |