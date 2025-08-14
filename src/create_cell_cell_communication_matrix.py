import argparse
import os
import re
import pandas as pd
import anndata as ad
import liana as li

def parse_args():
    parser = argparse.ArgumentParser(description='Generate cell-cell communication matrices using LIANA')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for communication matrices')
    parser.add_argument('--groupby', type=str, default='celltype',
                        help='Column name for cell type grouping (default: celltype)')
    parser.add_argument('--stage_key', type=str, default='day',
                        help='Column name for stage/time grouping (default: day)')
    parser.add_argument('--resource_name', type=str, default='consensus',
                        help='LIANA resource to use (default: consensus)')
    parser.add_argument('--use_raw', default=False, action='store_true',
                        help='Use raw counts from AnnData .raw slot')
    parser.add_argument('--verbose', default=True,  action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()

def extract_symbol(name):
    """Extract gene symbol from complex var_name string."""
    match = re.search(r'([A-Z0-9\-]+)(?:\[.*?\])?\s*\|', name)
    if match:
        return match.group(1)
    match = re.match(r'([A-Z0-9\-]+)', name)
    if match:
        return match.group(1)
    return None

def preprocess_gene_names(adata):
    """Extract gene symbols and filter AnnData."""
    symbols = [extract_symbol(x) for x in adata.var_names]
    adata.var['gene_symbol'] = symbols
    adata = adata[:, [s is not None and s != 'nan' for s in symbols]]
    adata.var_names = adata.var['gene_symbol']
    return adata

def get_communication_matrix(adata, all_celltypes):
    """Generate cell-cell communication matrix from LIANA results."""
    communication_dict = {}
    liana_df = adata.uns['liana_res']
    
    for celltype in all_celltypes:
        communication_dict[celltype] = []
        for celltype_2 in all_celltypes:
            if (celltype in liana_df['source'].unique() and 
                celltype_2 in liana_df['target'].unique()):
                score = liana_df[
                    (liana_df['source'] == celltype) & 
                    (liana_df['target'] == celltype_2)
                ]['lrscore'].values.sum()
                communication_dict[celltype].append(score)
            else:
                communication_dict[celltype].append(0)
    
    cc_comm_df = pd.DataFrame(communication_dict, 
                              index=all_celltypes, 
                              columns=all_celltypes)
    return cc_comm_df

def extract_gse_id(filename):
    """Extract GSE identifier from filename."""
    import re
    match = re.search(r'(GSE\d+)', filename)
    return match.group(1) if match else 'unknown'

def main():
    args = parse_args()
    
    # Extract GSE ID from input filename
    gse_id = extract_gse_id(os.path.basename(args.input_file))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    print(f"Loading data from {args.input_file}")
    scRNA = ad.read_h5ad(args.input_file)
    scRNA = preprocess_gene_names(scRNA)
    
    # Run LIANA on all data
    print("Running LIANA on all data...")
    li.method.rank_aggregate(
        adata=scRNA,
        groupby=args.groupby,
        resource_name=args.resource_name,
        use_raw=args.use_raw,
        verbose=args.verbose
    )
    
    # Get all cell types
    all_celltypes = scRNA.obs[args.groupby].unique()
    print(f"Found {len(all_celltypes)} cell types: {list(all_celltypes)}")
    
    # Generate communication matrix for all data
    print("Generating communication matrix for all data...")
    all_data_comm = get_communication_matrix(scRNA, all_celltypes)
    all_data_comm.to_csv(os.path.join(args.output_dir, f'all_at_once_{gse_id}.csv'))
    
    # Process by stages
    if args.stage_key in scRNA.obs.columns:
        stages = sorted(scRNA.obs[args.stage_key].unique())
        print(f"Processing {len(stages)} stages: {list(stages)}")
        
        scRNA_by_stage = []
        for stage in stages:
            print(f"Processing stage {stage}...")
            adata_stage = scRNA[scRNA.obs[args.stage_key] == stage].copy()
            
            li.method.rank_aggregate(
                adata=adata_stage,
                groupby=args.groupby,
                resource_name=args.resource_name,
                use_raw=args.use_raw,
                verbose=args.verbose
            )
            
            scRNA_by_stage.append(adata_stage)
            
            # Generate and save stage-specific communication matrix
            stage_comm = get_communication_matrix(adata_stage, all_celltypes)
            stage_comm.to_csv(
                os.path.join(args.output_dir, f'step_by_step_{stage}_{gse_id}.csv')
            )
        
        print(f"Saved communication matrices for all stages in {args.output_dir}")
    else:
        print(f"Warning: Stage key '{args.stage_key}' not found in data")
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()


"""

example usage:
note the directory structure of the input file and output directory

python create_cell_cell_communication_matrix.py \
    --input_file /Users/rssantanu/Desktop/codebase/constrained_FM/datasets/h5ad_processed_datasets/GSE232025_stereoseq.h5ad \
    --output_dir /Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_cell_communication_GSE232025 \
    --groupby celltype \
    --stage_key day \
    --resource_name consensus \
    --verbose

"""