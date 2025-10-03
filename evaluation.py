import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import os
import squidpy as sq
from sklearn.preprocessing import LabelEncoder
import json

import torch
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm
import pickle
from src.ode_helper_functions import *
from src.data_loading import preprocess_data, process_data, get_batch, get_batch_interpolation
from src.plots import plot_trajectories, plot_trajectories_new
from src.evaluate import *

from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP, GradModel, Graph_like_transformer
from torchcfm.utils import torch_wrapper

from sklearn.model_selection import train_test_split, KFold
import argparse

MODEL_REGISTRY = {
    "mlp": MLP,
    "graph_like_transformer": Graph_like_transformer,
    "gradmodel": GradModel,
    # Add more models here as needed
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CFM model with configurable parameters.")
    parser.add_argument("--train_config", type=str, default="v1", help="Name of the training config (without .json)")
    parser.add_argument("--h5ad_path", type=str, default="GSE232025_stereoseq.h5ad", help="Path to input h5ad file")
    parser.add_argument("--n_folds", type=int, default=1, help="Number of folds for cross-validation")
    parser.add_argument("--use_all_data", type=bool, default=False, help="Use all data for training")
    parser.add_argument("--new_experiment", type=str, default=None, help="there's several design choices for the experiment")
    parser.add_argument("--interpolation", type=bool, default=False, help="Use interpolation for training")
    parser.add_argument('--train_idx', nargs='+', type=int, help='List of training indices')
    parser.add_argument('--test_idx', nargs='+', type=int, help='List of test indices')
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size")
    args = parser.parse_args()

    num_folds = args.n_folds
    use_all_data = args.use_all_data    

    base_folder = f'/Users/rssantanu/Desktop/codebase/constrained_FM/experiment_figures/use_all_data_{args.use_all_data}_{args.h5ad_path.split("_")[0]}/{args.new_experiment}/'
    
    all_configs = os.listdir(base_folder)

    saved_configs = [c for c in all_configs if c.startswith(args.train_config)]
    for saved_config in saved_configs:
        fig_folder_address = base_folder + saved_config + '/'
        # IVP_dict = json.load(open(folder_address + 'IVP_error.json'))
        # next_step_dict = json.load(open(folder_address + 'next_step_error.json'))

        scRNA = ad.read_h5ad(os.path.join(os.path.dirname(__file__), 'datasets/h5ad_processed_datasets', args.h5ad_path))
        total_times = len(scRNA.obs['day'].unique())
        cell_type_key = 'celltype'

        params = json.load(open(f'train_configs/post_prior_correction/{args.train_config}.json'))

        # add more params keys
        params['dim'] = scRNA.obsm[f'X_{params["representation"]}'].shape[1]
        params['out_dim'] = scRNA.obsm[f'X_{params["representation"]}'].shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        day_list = (scRNA.obs["day"]).values.tolist()

        X_raw, data_df = preprocess_data(scRNA)
        X_phate, X_phate_conditional, Spatial, Celltype_list, microenvironment_features, LR_features = process_data(scRNA, X_raw, cell_type_key, total_times, params['use_spatial'], params['use_celltype_conditional'], params['use_bio_prior'], params['representation'])
        for i,j in zip(X_phate, X_raw):
            print(i.shape, j.shape)
        # import pdb; pdb.set_trace()
        multi_class_clf= pickle.load(open(f'/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_label_encoder_{args.h5ad_path.split("_")[0]}/multi_class_clf.pkl', 'rb'))
        label_encoder = pickle.load(open(f'/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_label_encoder_{args.h5ad_path.split("_")[0]}/label_encoder.pkl', 'rb'))
        
        for experiment_idx in range(num_folds):
            if args.train_idx is None and args.test_idx is None:
                train_idx, test_idx = train_test_split(np.arange(len(X_phate))[1:], test_size=args.test_size, random_state=44) # for this it's 123
            else:
                train_idx = args.train_idx
                test_idx = args.test_idx
            train_idx, test_idx = sorted(train_idx), sorted(test_idx)

            if use_all_data:
                train_idx = np.arange(len(X_phate))
                test_idx = []

            # always include the first timepoint in the training set
            X_phate_train= [X_phate[i] for i in [0]+train_idx]
            X_phate_conditional_train = [X_phate_conditional[i] for i in [0]+train_idx]

            X_phate_test= [X_phate[i] for i in test_idx]
            X_phate_conditional_test = [X_phate_conditional[i] for i in test_idx]


            
            Spatial_train = [Spatial[i] for i in [0]+train_idx]
            Spatial_test = [Spatial[i] for i in test_idx]
            
            Celltype_list_train = [Celltype_list[i] for i in [0]+train_idx]
            Celltype_list_test = [Celltype_list[i] for i in test_idx]

            microenvironment_features_train = [microenvironment_features[i] for i in [0]+train_idx]
            microenvironment_features_test = [microenvironment_features[i] for i in test_idx]

            LR_features_train = [LR_features[i] for i in [0]+train_idx]
            LR_features_test = [LR_features[i] for i in test_idx]
            
            
            n_times = len(X_phate_train)

            if params['use_celltype_conditional']:
                cond_dim = 1
            else:
                cond_dim = 0


            # ot_cfm_model = torch.load(f'{fig_folder_address}/neural_ode_model.pth')

            model_class = MODEL_REGISTRY.get(params['cfm_model'].lower())
            ot_cfm_model = model_class(
                dim=(params['dim']+1 if params['use_celltype_conditional'] else params['dim']),
                out_dim=params['out_dim'],
                time_varying=params['time_varying'],
                w=params['w']
            ).to(device)

            ot_cfm_model.load_state_dict(torch.load(f'{fig_folder_address}/neural_ode_model.pth'))
            node = NeuralODE(ConditionalODEVectorField(ot_cfm_model, cond_dim, params['dim']), solver="dopri5", sensitivity="adjoint")
            init_size= X_phate_train[0].shape[0]
            with torch.no_grad():
                if params['use_celltype_conditional']:
                    cfm_input = torch.cat([
                        torch.from_numpy(X_phate_train[0][:init_size]).float().to(device),
                        torch.from_numpy(X_phate_conditional_train[0][:init_size][:, None]).float().to(device)
                    ], dim=-1)
                else:
                    cfm_input = torch.from_numpy(X_phate_train[0][:init_size]).float().to(device)
                    
                traj = node.trajectory(
                    cfm_input,
                    t_span=torch.linspace(0, len(X_phate_train)+len(X_phate_test) - 1, params['trajectory_steps']),
                ).cpu()

            # because Celltype_list is empty when the use_celltype_conditional is false
            ref_idx= [len(X_phate[i]) for i in range(len(X_phate))]
            ref_idx= [sum(ref_idx[:i]) for i in range(len(ref_idx)+1)]
            gt_labels = scRNA.obs['celltype'].values.tolist()
            gt_labels = [gt_labels[ref_idx[i]:ref_idx[i+1]] for i in range(len(ref_idx)-1)]
            weighted_wasserstein_data = {'clf_model': multi_class_clf, 'gt_labels': gt_labels, 'label_encoder': label_encoder}
        
            try:
                mmd_list_next_step, wassersten_list_next_step, energy_list_next_step, r2_list_next_step, weighted_wasserstein_list_next_step, _, _ = evaluate_next_step(node, X_phate, device, n_points= 100000, weighted_wasserstein_data=weighted_wasserstein_data)
                metric_dict = {'mmd': mmd_list_next_step, 'wasserstein': wassersten_list_next_step, 'energy': energy_list_next_step, 'r2': r2_list_next_step, 'weighted_wasserstein': weighted_wasserstein_list_next_step}
                with open(f'{fig_folder_address}/next_step_error_exhaustive.json', 'w') as f:
                    json.dump(metric_dict, f)
            except:
                pass
            
            
            mmd_list_IVP, wassersten_list_IVP, energy_list_IVP, r2_list_IVP, weighted_wasserstein_list_IVP, _, _ = evaluate_IVP(traj, X_phate, device, weighted_wasserstein_data=weighted_wasserstein_data)
            metric_dict_IVP = {'mmd': mmd_list_IVP, 'wasserstein': wassersten_list_IVP, 'energy': energy_list_IVP, 'r2': r2_list_IVP, 'weighted_wasserstein': weighted_wasserstein_list_IVP}
            with open(f'{fig_folder_address}/IVP_error_exhaustive.json', 'w') as f:
                json.dump(metric_dict_IVP, f)