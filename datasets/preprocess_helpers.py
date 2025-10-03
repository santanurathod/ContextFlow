
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import os
import numpy as np
import pandas as pd
import squidpy as sq
import os
from sklearn.preprocessing import LabelEncoder
import scprep
from scvi.model import SCVI
import torch
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm
from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP, GradModel
from torchcfm.utils import plot_trajectories, torch_wrapper
import pandas as pd
import seaborn as sns
from umap import UMAP
import torch.nn as nn
from torchdyn.core import NeuralODE
import seaborn as sns
from scvi.model import SCVI
import scvi
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import liana as li
import re


def compute_local_mean(scRNA, representation='X_pca', spatial_key='spatial', radius=50):
    """
    Compute mean vector of the chosen representation for spatial neighbors.
    
    Parameters:
        scRNA: AnnData object
        representation: str, key in .obsm, e.g. 'X_pca', 'X_umap', 'X_scVI'
        spatial_key: str, key in .obsm with spatial coordinates
        radius: float, neighborhood radius in same units as coordinates
    """
    coords = scRNA.obsm[spatial_key]
    X = scRNA.obsm[representation]

    nbrs = NearestNeighbors(radius=radius).fit(coords)
    neighbors_idx = nbrs.radius_neighbors(coords, return_distance=False)

    local_means = np.zeros_like(X)

    for i, idx in enumerate(neighbors_idx):
        if len(idx) > 0:
            local_means[i] = X[idx].mean(axis=0)
        else:
            local_means[i] = X[i]  

    # Store the result in .obsm
    scRNA.obsm[f"local_mean_{representation}"] = local_means

    return scRNA


def get_scVI_latent_representation(scRNA, cell_type_key, spatial_key):
        print ("Using scVI for input data")
        #read the data (again cause scVI requires unormalized data) and set up scVI
        le = LabelEncoder()
        scRNA.obs[cell_type_key] = le.fit_transform(scRNA.obs[cell_type_key])
        scvi.model.SCVI.setup_anndata(scRNA)
        model = SCVI(scRNA)
        model.train()
        latent = model.get_latent_representation()
        scRNA.obsm["X_scVI"] = latent
        scRNA = compute_local_mean(scRNA, representation='X_scVI', radius=50)
        return scRNA



def get_LR_pattern_representation(scRNA, nz_prop=0.1, resource_name='consensus'):

    li.ut.spatial_neighbors(scRNA, bandwidth=200, cutoff=0.1, kernel='gaussian', set_diag=True)
    lrdata_by_day= []
    for stage in scRNA.obs['day'].unique():
        adata_stage = scRNA[scRNA.obs['day'] == stage].copy()
        lrdata = li.mt.bivariate(adata_stage,
                    resource_name=resource_name, # NOTE: uses HUMAN gene symbols!
                    local_name='cosine', # Name of the function
                    global_name="morans", # Name global function
                    n_perms=100, # Number of permutations to calculate a p-value
                    mask_negatives=False, # Whether to mask LowLow/NegativeNegative interactions
                    add_categories=True, # Whether to add local categories to the results
                    nz_prop=nz_prop, # Minimum expr. proportion for ligands/receptors and their subunits
                    use_raw=False,
                    verbose=True
                    )
        # Save or analyze results for this stage
        # adata_stage.uns['liana_res'].to_csv(f'liana_results_stage_{stage}.csv')
        lrdata_by_day.append(lrdata)


    

    # Step 1: Collect all unique LR pairs across all time points
    all_lr_pairs = set()
    for lrdata in lrdata_by_day:
        all_lr_pairs.update(lrdata.var_names)

    # Convert to sorted list for consistent ordering
    all_lr_pairs = sorted(list(all_lr_pairs))
    print(f"Total unique LR pairs across all time points: {len(all_lr_pairs)}")

    # Step 2: Create LR_pattern feature vectors for each time point
    for i, stage in enumerate(scRNA.obs['day'].unique()):
        # Get the subset of cells for this stage
        stage_mask = scRNA.obs['day'] == stage
        n_cells_stage = stage_mask.sum()
        
        # Initialize LR_pattern matrix with zeros
        lr_pattern_matrix = np.zeros((n_cells_stage, len(all_lr_pairs)))
        
        # Get the lrdata for this stage
        lrdata = lrdata_by_day[i]
        
        # Convert sparse matrix to dense if needed
        if sparse.issparse(lrdata.X):
            lrdata_dense = lrdata.X.toarray()
        else:
            lrdata_dense = lrdata.X
        
        # Fill in the values for LR pairs that exist in this stage
        for j, lr_pair in enumerate(all_lr_pairs):
            if lr_pair in lrdata.var_names:
                # Get the index of this LR pair in the current lrdata
                lr_idx = lrdata.var_names.get_loc(lr_pair)
                # Copy the values for all cells
                lr_pattern_matrix[:, j] = lrdata_dense[:, lr_idx]
            # If LR pair doesn't exist in this stage, it remains 0 (already initialized)
        
        # Add LR_pattern to obsm for cells of this stage
        if 'LR_pattern' not in scRNA.obsm:
            # Initialize the full LR_pattern matrix for all cells
            scRNA.obsm['LR_pattern'] = np.zeros((scRNA.n_obs, len(all_lr_pairs)))
        
        # Fill in the values for this stage
        scRNA.obsm['LR_pattern'][stage_mask] = lr_pattern_matrix
        
        print(f"Stage {stage}: {n_cells_stage} cells, {len(lrdata.var_names)} LR pairs available")

    # Step 3: Add metadata about the LR pairs
    scRNA.uns['LR_pattern_pairs'] = all_lr_pairs
    print(f"LR_pattern matrix shape: {scRNA.obsm['LR_pattern'].shape}")
    print(f"LR pair names stored in scRNA.uns['LR_pattern_pairs']")

    return scRNA


def clean_genes_for_LR_GSE232025(scRNA):
    scRNA.var["gene_symbol_original"] = scRNA.var_names
    def clean_name(name):
        # keep leftmost alias before '|'
        primary = name.split("|")[0]
        # remove [hs], [nr], etc.
        primary = re.sub(r"\[.*?\]", "", primary)
        return primary.strip()
    scRNA.var_names = [clean_name(x) for x in scRNA.var_names]
    scRNA.var_names_make_unique()
    return scRNA


def clean_genes_for_LR_GSE062025(scRNA):
    scRNA.var["gene_symbol_original"] = scRNA.var_names
    def clean_name(name):
        # keep leftmost alias before '|'
        primary = name.split("|")[0]
        # remove [hs], [nr], etc.
        primary = re.sub(r"\[.*?\]", "", primary)
        return primary.strip()
    scRNA.var_names = [clean_name(x) for x in scRNA.var_names]
    scRNA.var_names_make_unique()
    return scRNA



def homology_for_GSE072025(scRNA):

    map_df = pd.read_csv('/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/orthodb/hd_human_ortholog.csv')

    # 1) Build a mapping from your var_names → human symbol
    vn = pd.Index(scRNA.var_names)

    if vn.str.startswith('FBgn').all():             # case: var_names are FBgn IDs
        key = 'FBgn'
    elif vn.isin(map_df['gene_symbol']).mean() > 0: # case: var_names are fly symbols
        key = 'gene_symbol'
    else:
        raise ValueError('scRNA.var_names must be FBgn or fly gene symbols present in the mapping CSV')

    m = (map_df
        .dropna(subset=[key, 'Human_gene'])
        .drop_duplicates(subset=[key])
        .set_index(key)['Human_gene']
        .to_dict())

    # 2) Keep only genes we can map, attach human symbols
    keep_mask = vn.map(m).notna()
    scRNA_h = scRNA[:, keep_mask].copy()
    scRNA_h.var['human_symbol'] = scRNA_h.var_names.map(m)

    scRNA_h.var.index= scRNA_h.var['human_symbol']
    scRNA_h.var['gene_symbol_original'] = scRNA_h.var.index

    scRNA_h.var_names = scRNA_h.var['human_symbol'].str.upper().values

    return scRNA_h



import scanpy as sc
import pandas as pd
import numpy as np
import os

def load_visium_data(h5_path, meta_path, sample_id=None, spot_size=300, 
                     barcode_pattern=None, coord_cols=['visium_coor_x', 'visium_coor_y'],
                     sep=",", plot_spatial=True):
    """
    Load and process 10X Visium spatial transcriptomics data with metadata.
    
    Parameters:
    -----------
    h5_path : str
        Path to the .h5 feature-barcode matrix file
    meta_path : str  
        Path to the metadata CSV/TSV file
    sample_id : str, optional
        Sample identifier to clean from barcodes (e.g., 't_24h_m1_')
    spot_size : int, default 300
        Spot size for spatial plotting
    barcode_pattern : str, optional
        Regex pattern to clean barcodes (if sample_id not provided)
    coord_cols : list, default ['visium_coor_x', 'visium_coor_y']
        Column names for spatial coordinates in metadata
    sep : str, default ","
        Separator for metadata file
    plot_spatial : bool, default True
        Whether to generate spatial plot
        
    Returns:
    --------
    adata : AnnData
        Processed AnnData object with spatial coordinates and metadata
    """
    
    # Load 10X data
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    
    # Load metadata
    meta = pd.read_csv(meta_path, sep=sep)
    print(f"Metadata columns: {list(meta.columns)}")
    
    # Clean barcodes if sample_id or pattern provided
    if sample_id:
        # Auto-detect barcode column (common names)
        barcode_col = None
        for col in ['Spot_barcode', 'barcode', 'Barcode', 'spot_barcode']:
            if col in meta.columns:
                barcode_col = col
                break
        
        if barcode_col is None:
            raise ValueError(f"Could not find barcode column in metadata. Available: {list(meta.columns)}")
            
        # Clean metadata barcodes
        meta["barcode_clean"] = meta[barcode_col].str.replace(f"^{sample_id}", "", regex=True)
        
    elif barcode_pattern:
        meta["barcode_clean"] = meta[barcode_col].str.replace(barcode_pattern, "", regex=True)
    else:
        # Use original barcode column
        barcode_col = 'Spot_barcode' if 'Spot_barcode' in meta.columns else meta.columns[0]
        meta["barcode_clean"] = meta[barcode_col]
    
    # Standardize adata barcodes (common 10X format)
    adata.obs.index = adata.obs.index.str.replace("-1", "_1")
    
    # Set clean barcode as metadata index
    meta = meta.set_index("barcode_clean")
    
    # Join metadata to adata
    adata.obs = adata.obs.join(meta, how="left")
    
    # Add spatial coordinates
    if all(col in adata.obs.columns for col in coord_cols):
        adata.obsm["spatial"] = adata.obs[coord_cols].to_numpy()
        
        # Remove spots without spatial coordinates
        before_filter = adata.n_obs
        adata = adata[~adata.obs[coord_cols[0]].isna()].copy()
        after_filter = adata.n_obs
        
        print(f"Filtered {before_filter - after_filter} spots without coordinates")
        print(f"Final data: {adata.n_obs} spots × {adata.n_vars} genes")
        
        # Generate spatial plot
        if plot_spatial:
            sc.pl.spatial(adata, spot_size=spot_size)
            
    else:
        print(f"Warning: Coordinate columns {coord_cols} not found in metadata")
        print(f"Available columns: {list(adata.obs.columns)}")
    
    return adata


# Usage examples:
def load_liver_visium_samples():
    """Load multiple liver Visium samples with consistent processing."""
    
    base_path = "/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/raw_datasets/GSE092025/"
    meta_path = os.path.join(base_path, "Visium_Meta_data.txt")
    
    samples = {
        '24h_m1': ('Visium_24h_m1_raw_feature_bc_matrix.h5', 't_24h_m1_'),
        '24h_m2': ('Visium_24h_m2_raw_feature_bc_matrix.h5', 't_24h_m2_'),
        '48h_m4': ('Visium_48h_m4_raw_feature_bc_matrix.h5', 't_48h_m4_'),
        '48h_m5': ('Visium_48h_m5_raw_feature_bc_matrix.h5', 't_48h_m5_'),
        '72h_m1': ('Visium_72h_m1_raw_feature_bc_matrix.h5', 't_72h_m1_'),
        '72h_m2': ('Visium_72h_m2_raw_feature_bc_matrix.h5', 't_72h_m2_'),
    }
    
    datasets = {}
    for sample_name, (h5_file, sample_id) in samples.items():
        h5_path = os.path.join(base_path, h5_file)
        if os.path.exists(h5_path):
            print(f"\nLoading {sample_name}...")
            adata = load_visium_data(
                h5_path=h5_path,
                meta_path=meta_path,
                sample_id=sample_id,
                plot_spatial=False  # Set to True if you want individual plots
            )
            adata.obs['sample'] = sample_name
            adata.obs['timepoint'] = sample_name.split('_')[0]
            datasets[sample_name] = adata
        else:
            print(f"File not found: {h5_path}")
    
    return datasets