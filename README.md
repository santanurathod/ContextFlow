# Constrained Flow Matching

This repository contains code and instructions for processing single-cell RNA-seq datasets and running constrained flow matching experiments.

---

## Step 1: Prepare Raw Data

```bash
cd datasets
mkdir raw_datasets
mkdir h5ad_processed_datasets
cd raw_datasets
mkdir GSE149457
mkdir GSE232025
```

- **Copy the raw data:**
  - Place the chicken hearts data in the `GSE149457` folder.
  - Place the stereoseq data in the `GSE232025` folder.
- **File naming:**
  - For Stereoseq, rename the raw files as:
    - `d0_spatial_scRNAseq.h5ad`
    - `d1_spatial_scRNAseq.h5ad`
    - `d2_spatial_scRNAseq.h5ad`
    - `d3_spatial_scRNAseq.h5ad`
    - `d4_spatial_scRNAseq.h5ad`

---

## Step 2: Process the Data

Run the following commands to process and generate the `.h5ad` files:

```bash
python datasets/process_data.py --GSE GSE149457 --name_suffix chicken_hearts
python datasets/process_data.py --GSE GSE232025 --name_suffix stereo_seq
```

---
## Step 2.1 [Optional]: Incase you want to use weighted wasserstein

Need to generate the multi-class label classifier first; here it's XGBOOST model

```bash
python create_celltype_encoder.py --h5ad_path GSE232025_stereoseq.h5ad
```

## Step 2.2 [Optional]: Incase we need Ligand-Receptor Interaction network biological prior

Need to create the communication matrix first.
** We have two variations:**
  - all_at_once: considers communication as a bulk for all the days
  - step_by_step: cell-cell communication for each day; between t-t+1, consider the LR_CC for day_{t}

```bash
python creat_cell_cell_communication_matrix.py \
    --input_file /Users/rssantanu/Desktop/codebase/constrained_FM/datasets/h5ad_processed_datasets/GSE232025_stereoseq.h5ad \
    --output_dir /Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_cell_communication_GSE232025 \
    --groupby celltype \
    --stage_key day \
    --resource_name consensus \
    --verbose
```


---
## Step 3: Run Experiments

- **Experiment configurations** are provided as JSON files in the `train_configs` folder.

### Run a single experiment (example):

```bash
python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad
```

### Run multiple experiments:

```bash
bash run_exp_latest.sh
```

---

## Notes

- **Data files:** Large data files are not tracked in this repository. Please follow the instructions above to prepare your own data folders.
- **Configuration:** You can modify or add experiment configurations in the `train_configs` directory.

---

## License

[MIT License](LICENSE)