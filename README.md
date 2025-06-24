# Constrained Flow Matching

This repository contains code and instructions for processing single-cell RNA-seq datasets and running constrained flow matching experiments.

---

## Step 1: Prepare Raw Data

```bash
cd datasets
mkdir raw_datasets
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

## Step 3: Run Experiments

- **Experiment configurations** are provided as JSON files in the `train_configs` folder.

### Run a single experiment (example):

```bash
python main.py --train_config v1 --h5ad_path GSE232025_stereoseq.h5ad
```

### Run multiple experiments:

```bash
bash run_exp.sh
```

---

## Notes

- **Data files:** Large data files are not tracked in this repository. Please follow the instructions above to prepare your own data folders.
- **Configuration:** You can modify or add experiment configurations in the `train_configs` directory.

---

## License

[MIT License](LICENSE)