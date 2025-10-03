import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


import torch
from torchdyn.core import NeuralODE

import pandas as pd


def preprocess_data(scRNA):

    # Display the data matrix
    data_matrix = scRNA.X
    # data_matrix= scRNA.obsm["X_umap"]
    
    ind_list= list(scRNA.obs.index)
    day_list= list(scRNA.obs["day"])
    df_index= [ind_list[i].split("-")[0]+'_'+day_list[i] for i in range(len(ind_list))]

    # Convert to DataFrame for better readability
    data_df = pd.DataFrame(data_matrix.toarray() if hasattr(data_matrix, "toarray") else data_matrix,
                        index=df_index,
                        columns=scRNA.var_names)

    n_genes= 2000
    # Sum the expression values for each gene across all cells
    gene_sums = data_df.sum(axis=0)

    # Get the top n_genes genes based on expression sums
    top_genes = gene_sums.nlargest(n_genes).index

    # Filter the DataFrame to only include the top 100 expressed genes
    data_df = data_df[top_genes]

    # Display the DataFrame
    data_df['day']= [d.split('_')[-1] for d in data_df.index]

    all_times= data_df['day'].unique()


    # Remove the 'day' column from the DataFrame
    data_df = data_df.drop(columns=['day'])

    X_raw = [np.array(data_df[data_df.index.str.contains(f"_{d}")]) for d in all_times]
    return X_raw, data_df



# def process_data(scRNA, X_raw, cell_type_key='celltype', n_times=None, use_spatial=False, use_celltype=False, use_bio_prior=None, representation="umap"):

#     X_phate=[]
#     X_phate_conditional=[]
#     Spatial=[]
#     microenvironment_features= []
#     LR_features= []
#     Celltype_list= []
#     centroids= []
#     le = LabelEncoder()
#     encoded_labels = le.fit_transform(scRNA.obs[cell_type_key].values)
#     all_celltypes= scRNA.obs[cell_type_key].values.tolist()

#     left_counter=0
#     right_counter=X_raw[0].shape[0]
#     for i in range(n_times):
#         if representation == "umap":
#             X_phate.append(scRNA.obsm["X_umap"][left_counter:right_counter])
#         elif representation == "pca":
#             X_phate.append(scRNA.obsm["X_pca"][left_counter:right_counter])
#             microenvironment_features.append(scRNA.obsm["local_mean_X_pca"][left_counter:right_counter])
#             LR_features.append(scRNA.obsm["LR_pattern"][left_counter:right_counter])
#         elif representation =="scvi":
#             X_phate.append(scRNA.obsm["X_scvi"][left_counter:right_counter])
#             microenvironment_features.append(scRNA.obsm["local_mean_X_scvi"][left_counter:right_counter])
#             LR_features.append(scRNA.obsm["LR_pattern"][left_counter:right_counter])

#         X_phate_conditional.append(encoded_labels[left_counter:right_counter])
        
#         Spatial.append(scRNA.obsm['spatial'][left_counter:right_counter])

#         # if use_bio_prior:
#             # Celltype_list.append(all_celltypes[left_counter:right_counter])

#         # Celltype_list.append(encoded_labels[left_counter:right_counter]) ## note: these are encoded labels, not the actual cell types

#         Celltype_list.append(all_celltypes[left_counter:right_counter])

#         # print(left_counter, right_counter, X_raw[i].shape)

#         if i<n_times-1:
#             left_counter=right_counter
#             right_counter=left_counter+X_raw[i+1].shape[0]
        
        
    
#     return X_phate, X_phate_conditional, Spatial, Celltype_list, microenvironment_features, LR_features

def process_data(scRNA, X_raw, cell_type_key='celltype', n_times=None, use_spatial=False, use_celltype=False, use_bio_prior=None, representation="umap"):

    X_phate=[]
    X_phate_conditional=[]
    Spatial=[]
    microenvironment_features= []
    LR_features= []
    Celltype_list= []
    centroids= []
    le = LabelEncoder()
    encoded_labels = le.fit_transform(scRNA.obs[cell_type_key].values)
    all_celltypes= scRNA.obs[cell_type_key].values.tolist()

    # Get unique days and sort them to ensure consistent ordering
    all_times = sorted(scRNA.obs['day'].unique())
    
    for i, day in enumerate(all_times):
        # Create boolean mask for cells from this day
        day_mask = scRNA.obs['day'] == day
        day_indices = np.where(day_mask)[0]
        
        if representation == "umap":
            X_phate.append(scRNA.obsm["X_umap"][day_indices])
        elif representation == "pca":
            X_phate.append(scRNA.obsm["X_pca"][day_indices])
            microenvironment_features.append(scRNA.obsm["local_mean_X_pca"][day_indices])
            LR_features.append(scRNA.obsm["LR_pattern"][day_indices])
        elif representation =="scvi":
            X_phate.append(scRNA.obsm["X_scvi"][day_indices])
            microenvironment_features.append(scRNA.obsm["local_mean_X_scvi"][day_indices])
            LR_features.append(scRNA.obsm["LR_pattern"][day_indices])

        X_phate_conditional.append(encoded_labels[day_indices])
        Spatial.append(scRNA.obsm['spatial'][day_indices])
        Celltype_list.append([all_celltypes[idx] for idx in day_indices])

        print(f"Day {day}: {len(day_indices)} cells, X_raw[{i}]: {X_raw[i].shape}")
        
    return X_phate, X_phate_conditional, Spatial, Celltype_list, microenvironment_features, LR_features

def get_batch(FM, X, X_conditional, batch_size, n_times, return_noise=False, Spatial=[], Celltype_list=[], microenvironment_features=[], LR_features=[], device=None, params=None):
    """Construct a batch with point sfrom each timepoint pair"""
    ts = []
    xts = []
    xts_conditional=[]
    uts = []
    noises = []
    np.random.seed(42)
 
    for t_start in range(n_times - 1):

        try:
            b0= np.random.randint(X[t_start].shape[0], size=batch_size)
            b1= np.random.randint(X[t_start+1].shape[0], size=batch_size)
        except:
            import pdb; pdb.set_trace()


        # import pdb; pdb.set_trace()
        x0 = (torch.from_numpy(X[t_start][b0]).float().to(device))
        x0_conditional= (torch.from_numpy(X_conditional[t_start][b0]).float().to(device))

        if len(Celltype_list) > 0:
            ct0= np.array(Celltype_list[t_start])[b0]
            ct1= np.array(Celltype_list[t_start+1])[b1]
        else:
            ct0= None
            ct1= None
        
        # notice that it's shape[0] here
        x1 = (
            torch.from_numpy(
                X[t_start + 1][b1]
            )
            .float()
            .to(device)
        )
        x1_conditional= (
            torch.from_numpy(X_conditional[t_start+1][b1])
        ).float().to(device)
        
        if len(Spatial) > 0:
            p0= torch.from_numpy(Spatial[t_start][b0]).float().to(device)
            p1= torch.from_numpy(Spatial[t_start+1][b1]).float().to(device)
        else:
            p0= None
            p1= None

        if len(microenvironment_features) > 0:
            mc0= torch.from_numpy(microenvironment_features[t_start][b0]).float().to(device)
            mc1= torch.from_numpy(microenvironment_features[t_start+1][b1]).float().to(device)
        else:
            mc0= None
            mc1= None
            
        if len(LR_features) > 0:
            lr0= torch.from_numpy(LR_features[t_start][b0]).float().to(device)
            lr1= torch.from_numpy(LR_features[t_start+1][b1]).float().to(device)
        else:
            lr0= None
            lr1= None

        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, p0, p1, ct0, ct1, mc0, mc1, lr0, lr1, return_noise=return_noise, cc_index=t_start, params=params)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
        xts_conditional.append(x0_conditional)


    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    xts_conditional= torch.cat(xts_conditional)
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises, xts_conditional
    return t, xt, ut, xts_conditional



def get_batch_interpolation(FM, X, X_conditional, batch_size, n_times, return_noise=False, Spatial=[], Celltype_list=[], microenvironment_features=[], LR_features=[], device=None, params=None, train_idx=None):
    """Construct a batch with point from each timepoint pair
    getting proper pairs of data, I made this for the interpolation case
    """
    ts = []
    xts = []
    xts_conditional=[]
    uts = []
    noises = []
    np.random.seed(42)

    for i in range(len(train_idx) - 1):
        t_start = train_idx[i]
        t_end = train_idx[i+1]
        try:
            b0= np.random.randint(X[t_start].shape[0], size=batch_size)
            b1= np.random.randint(X[t_end].shape[0], size=batch_size)
        except:
            import pdb; pdb.set_trace()


        # import pdb; pdb.set_trace()
        x0 = (torch.from_numpy(X[t_start][b0]).float().to(device))
        x0_conditional= (torch.from_numpy(X_conditional[t_start][b0]).float().to(device))

        if len(Celltype_list) > 0:
            ct0= np.array(Celltype_list[t_start])[b0]
            ct1= np.array(Celltype_list[t_end])[b1]
        else:
            ct0= None
            ct1= None
        
        # notice that it's shape[0] here
        x1 = (
            torch.from_numpy(
                X[t_end][b1]
            )
            .float()
            .to(device)
        )
        x1_conditional= (
            torch.from_numpy(X_conditional[t_end][b1])
        ).float().to(device)
        
        if len(Spatial) > 0:
            p0= torch.from_numpy(Spatial[t_start][b0]).float().to(device)
            p1= torch.from_numpy(Spatial[t_end][b1]).float().to(device)
        else:
            p0= None
            p1= None

        if len(microenvironment_features) > 0:
            mc0= torch.from_numpy(microenvironment_features[t_start][b0]).float().to(device)
            mc1= torch.from_numpy(microenvironment_features[t_end][b1]).float().to(device)
        else:
            mc0= None
            mc1= None
            
        if len(LR_features) > 0:
            lr0= torch.from_numpy(LR_features[t_start][b0]).float().to(device)
            lr1= torch.from_numpy(LR_features[t_end][b1]).float().to(device)
        else:
            lr0= None
            lr1= None

        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow_interpolation(
                x0, x1, return_noise=return_noise, t_start=t_start, t_end=t_end
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow_interpolation(x0, x1, p0, p1, ct0, ct1, mc0, mc1, lr0, lr1, return_noise=return_noise, cc_index=t_start, params=params, t_start=t_start, t_end=t_end)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
        xts_conditional.append(x0_conditional)


    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    xts_conditional= torch.cat(xts_conditional)
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises, xts_conditional
    return t, xt, ut, xts_conditional