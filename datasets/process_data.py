import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import os
import numpy as np
import pandas as pd
import squidpy as sq

import seaborn as sns
import anndata as ad
import matplotlib.colors as mcolors
from preprocess_helpers import get_scVI_latent_representation, compute_local_mean, get_LR_pattern_representation, clean_genes_for_LR_GSE232025, clean_genes_for_LR_GSE062025, homology_for_GSE072025
from preprocess_helpers import load_liver_visium_samples

base_path= "/Users/rssantanu/Desktop/codebase/constrained_FM/datasets"

n_comps = 50

def load_scRNA_GSE149457(data_path):
    """
    This function processes the GSE149457 dataset and returns the concatenated AnnData object.
    """
    # Get the raw data

    sc1 = sc.read_10x_h5(os.path.join(data_path, "GSM4502475_chicken_heart_scRNAseq_D4_filtered_feature_bc_matrix.h5"))
    sc2 = sc.read_10x_h5(os.path.join(data_path, "GSM4502476_chicken_heart_scRNAseq_D7_LV_filtered_feature_bc_matrix.h5"))
    sc3 = sc.read_10x_h5(os.path.join(data_path, "GSM4502477_chicken_heart_scRNAseq_D7_RV_filtered_feature_bc_matrix.h5"))
    sc4 = sc.read_10x_h5(os.path.join(data_path, "GSM4502478_chicken_heart_scRNAseq_D10_LV_filtered_feature_bc_matrix.h5"))
    sc5 = sc.read_10x_h5(os.path.join(data_path, "GSM4502479_chicken_heart_scRNAseq_D10_RV_filtered_feature_bc_matrix.h5"))
    sc6 = sc.read_10x_h5(os.path.join(data_path, "GSM4502480_chicken_heart_scRNAseq_D14_LV_filtered_feature_bc_matrix.h5"))
    sc7 = sc.read_10x_h5(os.path.join(data_path, "GSM4502481_chicken_heart_scRNAseq_D14_RV_filtered_feature_bc_matrix.h5"))
    meta = pd.read_csv(os.path.join(data_path, "scRNAseq_metadata.csv"), index_col=0)
    sc1.var_names_make_unique()
    sc2.var_names_make_unique()
    sc3.var_names_make_unique()
    sc4.var_names_make_unique()
    sc5.var_names_make_unique()
    sc6.var_names_make_unique()
    sc7.var_names_make_unique()
    scRNA = sc1.concatenate(sc2, sc3, sc4, sc5, sc6, sc7)

    ## Process the metadata because of data inconsistency
    meta = meta.reset_index()
    # Rename for clarity
    meta = meta.rename(columns={"index": "original_id"})
    # Split into sample and barcode based on the first underscore
    meta["sample"] = meta["original_id"].str.extract(r"^(.*?)(?=_)")      
    meta["barcode"] = meta["original_id"].str.extract(r"^[^_]+_(.*)$")

    batch_map = {"D4": "0", "D7-LV": "1", "D7-RV": "2", "D10-LV": "3", "D10-RV": "4", "D14-LV": "5", "D14-RV": "6"}
   
    meta["Batch"] = meta["sample"].map(batch_map)
    meta["combined"] = meta["barcode"] + "-1-" + meta["Batch"]

    meta.set_index("combined", inplace=True)
    meta = meta.drop(columns=["Batch"])
    meta = meta.loc[scRNA.obs_names.intersection(meta.index)]
    # Join into AnnData object
    scRNA.obs = scRNA.obs.join(meta)
    scRNA.obs = scRNA.obs.rename(columns={"celltypes.0.5": "celltype"})
    scRNA = scRNA[scRNA.obs["celltype"].notna()]


    return scRNA
    

def load_scRNA_GSE232025(data_path):
    """
    This function processes the GSE232025 dataset and returns the concatenated AnnData object.
    """
    sc1= sc.read_h5ad(os.path.join(data_path, "d0_spatial_scRNAseq.h5ad"))
    sc1.obs["Batch"] = "0"
    sc2= sc.read_h5ad(os.path.join(data_path, "d1_spatial_scRNAseq.h5ad"))
    sc2.obs["Batch"] = "1"
    sc3= sc.read_h5ad(os.path.join(data_path, "d2_spatial_scRNAseq.h5ad"))
    sc3.obs["Batch"] = "2"
    sc4= sc.read_h5ad(os.path.join(data_path, "d3_spatial_scRNAseq.h5ad"))
    sc4.obs["Batch"] = "3"
    sc5= sc.read_h5ad(os.path.join(data_path, "d4_spatial_scRNAseq.h5ad"))
    sc5.obs["Batch"] = "4"

    adatas = [sc1, sc2, sc3, sc4, sc5]

    ## cell color mapping
    all_cell_types = set()
    for adata in adatas:
        all_cell_types.update(adata.obs["Annotation"].unique())

    # Create a consistent color map for all cell types
    palette = sns.color_palette("tab20", len(all_cell_types))
    color_dict = dict(zip(sorted(all_cell_types), palette))

    for sci in adatas:
        sci.obs = sci.obs.rename(columns={"Annotation": "celltype", "Batch": "day"})
        sci.obs["celltype"] = sci.obs["celltype"].astype(str)
        sci.obs["color"] = sci.obs["celltype"].map(color_dict)
        sci.obs["color"] = sci.obs["color"].apply(lambda x: mcolors.to_hex(x) if isinstance(x, (tuple, list, np.ndarray)) else x)
    scRNA = ad.concat(adatas, label='day', keys=[a.obs['day'][0] for a in adatas])

    return scRNA


def load_scRNA_GSE062025(data_path):
    """
    This function processes the GSE062025 dataset and returns the concatenated AnnData object.
    """
    sc1= sc.read_h5ad(os.path.join(data_path, "E9.5.h5ad"))
    sc1.obs["Batch"] = "0"
    sc2= sc.read_h5ad(os.path.join(data_path, "E10.5.h5ad"))
    sc2.obs["Batch"] = "1"
    sc3= sc.read_h5ad(os.path.join(data_path, "E11.5.h5ad"))
    sc3.obs["Batch"] = "2"
    sc4= sc.read_h5ad(os.path.join(data_path, "E12.5.h5ad"))
    sc4.obs["Batch"] = "3"
    sc5= sc.read_h5ad(os.path.join(data_path, "E13.5.h5ad"))
    sc5.obs["Batch"] = "4"
    sc6= sc.read_h5ad(os.path.join(data_path, "E14.5.h5ad"))
    sc6.obs["Batch"] = "5"
    sc7= sc.read_h5ad(os.path.join(data_path, "E15.5.h5ad"))
    sc7.obs["Batch"] = "6"
    sc8= sc.read_h5ad(os.path.join(data_path, "E16.5.h5ad"))
    sc8.obs["Batch"] = "7"

    adatas = [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8]
    # adatas = [sc1, sc2, sc3, sc4, sc5, sc7, sc8]
    ## cell color mapping
    all_cell_types = set()
    for adata in adatas:
        all_cell_types.update(adata.obs["annotation"].unique())

    # Create a consistent color map for all cell types
    palette = sns.color_palette("tab20", len(all_cell_types))
    color_dict = dict(zip(sorted(all_cell_types), palette))

    for sci in adatas:
        sci.obs = sci.obs.rename(columns={"annotation": "celltype", "Batch": "day"})
        sci.obs["celltype"] = sci.obs["celltype"].astype(str)
        sci.obs["color"] = sci.obs["celltype"].map(color_dict)
        sci.obs["color"] = sci.obs["color"].apply(lambda x: mcolors.to_hex(x) if isinstance(x, (tuple, list, np.ndarray)) else x)
    scRNA = ad.concat(adatas, label='day', keys=[a.obs['day'][0] for a in adatas])

    return scRNA


def load_scRNA_GSE072025(data_path):
    """
    This function processes the GSE072025 dataset and returns the concatenated AnnData object.
    """
    sc1= sc.read_h5ad(os.path.join(data_path, "E15.h5ad"))
    sc1.obs["Batch"] = "0"
    sc2= sc.read_h5ad(os.path.join(data_path, "E16.h5ad"))
    sc2.obs["Batch"] = "1"
    sc3= sc.read_h5ad(os.path.join(data_path, "E17.h5ad"))
    sc3.obs["Batch"] = "2"
    sc4= sc.read_h5ad(os.path.join(data_path, "E18.h5ad"))
    sc4.obs["Batch"] = "3"
    sc5= sc.read_h5ad(os.path.join(data_path, "E19.h5ad"))
    sc5.obs["Batch"] = "4"

    adatas = [sc1, sc2, sc3, sc4, sc5]

    ## cell color mapping
    all_cell_types = set()
    for adata in adatas:
        all_cell_types.update(adata.obs["annotation"].unique())

    # Create a consistent color map for all cell types
    palette = sns.color_palette("tab20", len(all_cell_types))
    color_dict = dict(zip(sorted(all_cell_types), palette))

    for sci in adatas:
        sci.obs = sci.obs.rename(columns={"annotation": "celltype", "Batch": "day"})
        sci.obs["celltype"] = sci.obs["celltype"].astype(str)
        sci.obs["color"] = sci.obs["celltype"].map(color_dict)
        sci.obs["color"] = sci.obs["color"].apply(lambda x: mcolors.to_hex(x) if isinstance(x, (tuple, list, np.ndarray)) else x)
    scRNA = ad.concat(adatas, label='day', keys=[a.obs['day'][0] for a in adatas])

    return scRNA


def load_visium_GSE092025():
    datasets = load_liver_visium_samples()
    datasets['24h_m1'].obs['Batch'] = '0'
    datasets['24h_m2'].obs['Batch'] = '0'
    datasets['48h_m4'].obs['Batch'] = '1'
    datasets['48h_m5'].obs['Batch'] = '1'
    datasets['72h_m1'].obs['Batch'] = '2'
    datasets['72h_m2'].obs['Batch'] = '2'

    adatas = [datasets['24h_m1'], datasets['48h_m5'], datasets['72h_m1']]

    ## cell color mapping
    all_cell_types = set()
    for adata in adatas:
        all_cell_types.update(adata.obs["Fibrogenic_spot"].unique())

    # Create a consistent color map for all cell types
    palette = sns.color_palette("tab20", len(all_cell_types))
    color_dict = dict(zip(sorted(all_cell_types), palette))

    for sci in adatas:
        sci.obs = sci.obs.rename(columns={"Fibrogenic_spot": "celltype", "Batch": "day"})
        sci.obs["celltype"] = sci.obs["celltype"].astype(str)
        sci.obs["color"] = sci.obs["celltype"].map(color_dict)
        sci.obs["color"] = sci.obs["color"].apply(lambda x: mcolors.to_hex(x) if isinstance(x, (tuple, list, np.ndarray)) else x)
    scRNA = ad.concat(adatas, label='day', keys=[a.obs['day'][0] for a in adatas])


    return scRNA





def apply_processing_steps(scRNA, GSE_ID, num_genes=2000, n_comps=50, nz_prop=0.1):
    """
    This function applies the processing steps to the AnnData object.
    There needs to standardized processing steps for all datasets, this is a temporary solution.
    """
    if GSE_ID == "GSE232025":
        sc.pp.highly_variable_genes(scRNA, batch_key='day', n_top_genes=num_genes, flavor='seurat_v3')
        scRNA = scRNA[:, scRNA.var.highly_variable]
        sc.tl.pca(scRNA, n_comps=n_comps)
        sc.pp.neighbors(scRNA)
        sc.tl.umap(scRNA)
        scRNA= compute_local_mean(scRNA, representation='X_pca')
        # scRNA = get_scVI_latent_representation(scRNA, cell_type_key='celltype', spatial_key='spatial')

        scRNA = clean_genes_for_LR_GSE232025(scRNA)
        scRNA = get_LR_pattern_representation(scRNA, nz_prop=nz_prop)

    elif GSE_ID == "GSE149457":
        sc.pp.normalize_total(scRNA, target_sum=1e4)
        sc.pp.log1p(scRNA)
        sc.pp.highly_variable_genes(scRNA, n_top_genes=num_genes, subset=True)
        sc.pp.scale(scRNA, max_value=10)
        sc.tl.pca(scRNA, n_comps=n_comps)
        sc.pp.neighbors(scRNA)
        sc.tl.umap(scRNA)
    
    elif GSE_ID == "GSE062025":
        sc.pp.highly_variable_genes(scRNA, batch_key='day', n_top_genes=num_genes, flavor='seurat_v3')
        scRNA = scRNA[:, scRNA.var.highly_variable]
        sc.tl.pca(scRNA, n_comps=n_comps)
        sc.pp.neighbors(scRNA)
        sc.tl.umap(scRNA)
        scRNA= compute_local_mean(scRNA, representation='X_pca')
        # scRNA = get_scVI_latent_representation(scRNA, cell_type_key='celltype', spatial_key='spatial')

        scRNA = clean_genes_for_LR_GSE062025(scRNA)
        scRNA = get_LR_pattern_representation(scRNA, nz_prop=nz_prop, resource_name='MouseConsensus')
    
    elif GSE_ID == "GSE072025":
        sc.pp.highly_variable_genes(scRNA, batch_key='day', n_top_genes=num_genes, flavor='seurat_v3')
        scRNA = scRNA[:, scRNA.var.highly_variable]
        sc.tl.pca(scRNA, n_comps=n_comps)
        sc.pp.neighbors(scRNA)
        sc.tl.umap(scRNA)
        scRNA= compute_local_mean(scRNA, representation='X_pca')
        # scRNA = get_scVI_latent_representation(scRNA, cell_type_key='celltype', spatial_key='spatial')

        scRNA = homology_for_GSE072025(scRNA)
        scRNA = get_LR_pattern_representation(scRNA, nz_prop=nz_prop, resource_name='Consensus')
    
    elif GSE_ID == "GSE092025":
        sc.pp.highly_variable_genes(scRNA, batch_key='day', n_top_genes=num_genes, flavor='seurat_v3')
        scRNA = scRNA[:, scRNA.var.highly_variable]
        sc.tl.pca(scRNA, n_comps=n_comps)
        sc.pp.neighbors(scRNA)
        sc.tl.umap(scRNA)
        scRNA= compute_local_mean(scRNA, representation='X_pca')
        # scRNA = get_scVI_latent_representation(scRNA, cell_type_key='celltype', spatial_key='spatial')

        scRNA = get_LR_pattern_representation(scRNA, nz_prop=nz_prop)

    return scRNA

def get_concatenated_dynamic_data(GSE_ID, save_path=None, data_type="scRNA", num_genes=2000, n_comps=50, nz_prop=0.1):
    """
    This function concatenates the dynamic data from the raw datasets and saves it as a csv file.
    """
    # Get the raw data

    if GSE_ID == "GSE149457" and data_type == "scRNA":
        data_path = os.path.join(base_path, "raw_datasets", "GSE149457")
        scRNA = load_scRNA_GSE149457(data_path)
        scRNA = apply_processing_steps(scRNA, GSE_ID, num_genes=num_genes, n_comps=n_comps, nz_prop=nz_prop)
    
    elif GSE_ID == "GSE232025" and data_type == "scRNA":
        data_path = os.path.join(base_path, "raw_datasets", "GSE232025")
        scRNA = load_scRNA_GSE232025(data_path)
        scRNA = apply_processing_steps(scRNA, GSE_ID, num_genes=num_genes, n_comps=n_comps, nz_prop=nz_prop)
    
    elif GSE_ID == "GSE062025" and data_type == "scRNA":
        data_path = os.path.join(base_path, "raw_datasets", "GSE062025")
        scRNA = load_scRNA_GSE062025(data_path)
        scRNA = apply_processing_steps(scRNA, GSE_ID, num_genes=num_genes, n_comps=n_comps, nz_prop=nz_prop)
    
    elif GSE_ID == "GSE072025" and data_type == "scRNA":
        data_path = os.path.join(base_path, "raw_datasets", "GSE072025")
        scRNA = load_scRNA_GSE072025(data_path)
        scRNA = apply_processing_steps(scRNA, GSE_ID, num_genes=num_genes, n_comps=n_comps, nz_prop=nz_prop)
    
    elif GSE_ID == "GSE092025":
        scRNA = load_visium_GSE092025()
        scRNA = apply_processing_steps(scRNA, GSE_ID, num_genes=num_genes, n_comps=n_comps, nz_prop=nz_prop)
    
    return scRNA


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process scRNA-seq data for a given GSE ID.")
    parser.add_argument("--GSE", type=str, required=True, help="GSE ID to process (e.g., GSE232025 or GSE149457)")
    parser.add_argument("--name_suffix", type=str, required=True, help="Name suffix to add to the processed data (e.g., _processed)")
    parser.add_argument("--num_genes", type=int, required=False, default=2000, help="Number of genes to use for the processed data")
    parser.add_argument("--n_comps", type=int, required=False, default=50, help="Number of components to use for the processed data")
    parser.add_argument("--nz_prop", type=float, required=False, default=0.1, help="Number of components to use for the processed data")
    args = parser.parse_args()

    scRNA = get_concatenated_dynamic_data(GSE_ID=args.GSE, num_genes=args.num_genes, n_comps=args.n_comps, nz_prop=args.nz_prop)
    scRNA.write(os.path.join(base_path, 'h5ad_processed_datasets', f"{args.GSE}_{args.name_suffix}_g_{args.num_genes}_nzp_{args.nz_prop}.h5ad"))
