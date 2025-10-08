# Dataset Processing Pipeline

This guide explains how to download and preprocess spatial omics datasets for trajectory inference.

## 1. Download Raw Data

Use the download script to fetch datasets by their GSE accession number:

### Axolotl Brain Regeneration (GSE232025)
```bash
python ./raw_datasets/download_datasets.py --dataset GSE232025
```

### Mouse Organogenesis (GSE062025)
```bash
python ./raw_datasets/download_datasets.py --dataset GSE062025
```

### Liver Regeneration (GSE092025)
```bash
python ./raw_datasets/download_datasets.py --dataset GSE092025
```

## 2. Preprocess Data

Process the raw data and store as h5ad objects with the following parameters:

### Parameters

- `--GSE`: GSE accession number of the dataset
- `--name_suffix`: Dataset identifier suffix
- `--num_genes`: Number of most dynamical genes to consider (default: 1000)
- `--n_comps`: Number of PCA components (default: 50)
- `--nz_prop`: Minimum expression proportion for ligands/receptors and their subunits (LIANA+ hyperparameter)

### Preprocessing Commands

#### Axolotl Dataset (GSE232025)
```bash
python process_data.py --GSE GSE232025 --name_suffix stereoseq --num_genes 10000 --nz_prop 0.1
```

#### Mouse Dataset (GSE062025)
```bash
python process_data.py --GSE GSE062025 --name_suffix mosta --num_genes 10000 --nz_prop 0.1
```

#### Liver Regeneration Dataset (GSE092025)
```bash
python process_data.py --GSE GSE092025 --name_suffix liver --num_genes 10000 --nz_prop 0.06
```

> **Note**: For the liver dataset, `nz_prop` is set to 0.06 as higher values do not yield any ligand-receptor pairs.

## Notes

- **num_genes**: We use 10,000 genes to capture sufficient biological variation while filtering noise
- **nz_prop**: Dataset-specific tuning may be required to ensure adequate ligand-receptor pair detection
- All datasets use 50 PCA components by default for dimensionality reduction