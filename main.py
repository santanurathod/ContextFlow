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
from src.data_loading import preprocess_data, process_data, get_batch
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


def train(params, n_times, X_phate, X_phate_conditional, device, fig_address, Spatial=[], Celltype_list=[]):
    use_cuda = torch.cuda.is_available()
    batch_size = params['batch_size']
    sigma = params['sigma']

    model_class = MODEL_REGISTRY.get(params['cfm_model'].lower())
    if model_class is None:
        raise ValueError(f"Unknown model: {params['cfm_model']}. Available: {list(MODEL_REGISTRY.keys())}")
    
    ot_cfm_model = model_class(
        dim=(params['dim']+1 if params['use_celltype_conditional'] else params['dim']),
        out_dim=params['out_dim'],
        time_varying=params['time_varying'],
        w=params['w']
    ).to(device)

    ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), params['learning_rate'])
    if 'ot_method' in params and 'entropy_reg' in params:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=params['sigma'], ot_method=params['ot_method'], entropy_reg=params['entropy_reg'])
    else:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=params['sigma'])

    losses = []
    for i in tqdm(range(params['n_epochs'])):
        ot_cfm_optimizer.zero_grad()
        t, xt, ut, xt_conditional = get_batch(FM, X_phate, X_phate_conditional, batch_size, n_times, Spatial=Spatial, Celltype_list=Celltype_list, device=device, params=params)
        if params['use_celltype_conditional']:
            cfm_input = torch.cat([xt, xt_conditional[:, None], t[:, None]], dim=-1)
        else:
            cfm_input = torch.cat([xt, t[:, None]], dim=-1)
        
        vt = ot_cfm_model(cfm_input)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        losses.append(loss.item())
        ot_cfm_optimizer.step()

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(f"{fig_address}/loss_curve.png")
    plt.close()

    return ot_cfm_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CFM model with configurable parameters.")
    parser.add_argument("--train_config", type=str, default="v1", help="Name of the training config (without .json)")
    parser.add_argument("--h5ad_path", type=str, default="GSE232025_stereoseq.h5ad", help="Path to input h5ad file")
    parser.add_argument("--n_folds", type=int, default=1, help="Number of folds for cross-validation")
    parser.add_argument("--use_all_data", type=bool, default=False, help="Use all data for training")
    parser.add_argument("--new_experiment", type=str, default=None, help="there's several design choices for the experiment")
    args = parser.parse_args()

    num_folds = args.n_folds
    use_all_data = args.use_all_data
    if args.new_experiment is not None:
        fig_folder_address = f'./experiment_figures/use_all_data_{args.use_all_data}_{args.h5ad_path.split("_")[0]}/{args.new_experiment}/{args.train_config}'
    else:
        fig_folder_address = f'./experiment_figures/use_all_data_{args.use_all_data}_{args.h5ad_path.split("_")[0]}/{args.train_config}'
    curr_i = 0
    if os.path.exists(fig_folder_address):
        while os.path.exists(fig_folder_address):
            fig_folder_address = fig_folder_address + f'_{curr_i}'
            curr_i += 1
    os.makedirs(fig_folder_address, exist_ok=True)
    
    print(f'fig_folder_address: {fig_folder_address}')

    scRNA = ad.read_h5ad(os.path.join(os.path.dirname(__file__), 'datasets/h5ad_processed_datasets', args.h5ad_path))
    labels_dict = {'1':1, '2':2, '3':3, '4':4, '5':5}
    cell_type_key = 'celltype'

    params = json.load(open(f'train_configs/{args.train_config}.json'))

    # add more params keys
    params['dim'] = scRNA.obsm[f'X_{params["representation"]}'].shape[1]
    params['out_dim'] = scRNA.obsm[f'X_{params["representation"]}'].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    day_list = (scRNA.obs["day"]).values.tolist()

    X_raw, data_df = preprocess_data(scRNA)
    X_phate, X_phate_conditional, Spatial, Celltype_list = process_data(scRNA, X_raw, cell_type_key, labels_dict, params['use_spatial'], params['use_celltype_conditional'], params['use_bio_prior'], params['representation'])

    multi_class_clf= pickle.load(open(f'/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_label_encoder_{args.h5ad_path.split("_")[0]}/multi_class_clf.pkl', 'rb'))
    label_encoder = pickle.load(open(f'/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_label_encoder_{args.h5ad_path.split("_")[0]}/label_encoder.pkl', 'rb'))
    
    for experiment_idx in range(num_folds):
        train_idx, test_idx = train_test_split(np.arange(len(X_phate))[1:], test_size=0.2, random_state=44) # for this it's 123
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
        
        
        n_times = len(X_phate_train)

        ot_cfm_model = train(params, n_times, X_phate_train, X_phate_conditional_train, device, fig_folder_address, Spatial_train, Celltype_list_train)

        if params['use_celltype_conditional']:
            cond_dim = 1
        else:
            cond_dim = 0
        node = NeuralODE(ConditionalODEVectorField(ot_cfm_model, cond_dim, params['dim']), solver="dopri5", sensitivity="adjoint")
        with torch.no_grad():
            if params['use_celltype_conditional']:
                cfm_input = torch.cat([
                    torch.from_numpy(X_phate_train[0][:1000]).float().to(device),
                    torch.from_numpy(X_phate_conditional_train[0][:1000][:, None]).float().to(device)
                ], dim=-1)
            else:
                cfm_input = torch.from_numpy(X_phate_train[0][:1000]).float().to(device)
                
            traj = node.trajectory(
                cfm_input,
                t_span=torch.linspace(0, len(X_phate_train)+len(X_phate_test) - 1, params['trajectory_steps']),
            ).cpu()

            traj_fig_address = f'{fig_folder_address}/trajectory.png'
            plot_trajectories(scRNA, traj.cpu().numpy(), day_list, traj_fig_address)

        # because Celltype_list is empty when the use_celltype_conditional is false
        ref_idx= [len(X_phate[i]) for i in range(len(X_phate))]
        ref_idx= [sum(ref_idx[:i]) for i in range(len(ref_idx)+1)]
        gt_labels = scRNA.obs['celltype'].values.tolist()
        gt_labels = [gt_labels[ref_idx[i]:ref_idx[i+1]] for i in range(len(ref_idx)-1)]
        weighted_wasserstein_data = {'clf_model': multi_class_clf, 'gt_labels': gt_labels, 'label_encoder': label_encoder}
    
        # import pdb; pdb.set_trace()

        try:
            mmd_list_next_step, wassersten_list_next_step, energy_list_next_step, r2_list_next_step, weighted_wasserstein_list_next_step = evaluate_next_step(node, X_phate, device, weighted_wasserstein_data=weighted_wasserstein_data)
            plt.figure(figsize=(8,6))
            plt.plot([1+i for i in range(len(mmd_list_next_step))], mmd_list_next_step, label='MMD', marker='o')
            plt.plot([1+i for i in range(len(wassersten_list_next_step))], wassersten_list_next_step, label='Wasserstein', marker='s')
            plt.plot([1+i for i in range(len(energy_list_next_step))], energy_list_next_step, label='Energy Distance', marker='^')
            plt.plot([1+i for i in range(len(r2_list_next_step))], r2_list_next_step, label='R2 Score', marker='D')
            plt.plot([1+i for i in range(len(weighted_wasserstein_list_next_step))], weighted_wasserstein_list_next_step, label='Weighted Wasserstein', marker='*')
            plt.xlabel("Timepoint t → t+1")
            plt.ylabel("Error")
            plt.title("Distribution Matching Error")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{fig_folder_address}/next_step_error.png')
            plt.close()

            metric_dict = {'mmd': mmd_list_next_step, 'wasserstein': wassersten_list_next_step, 'energy': energy_list_next_step, 'r2': r2_list_next_step, 'weighted_wasserstein': weighted_wasserstein_list_next_step}
            with open(f'{fig_folder_address}/next_step_error.json', 'w') as f:
                json.dump(metric_dict, f)
        except:
            pass
        
        
        mmd_list_IVP, wassersten_list_IVP, energy_list_IVP, r2_list_IVP, weighted_wasserstein_list_IVP = evaluate_IVP(traj, X_phate, device, weighted_wasserstein_data=weighted_wasserstein_data)

        plt.figure(figsize=(8,6))
        plt.plot(range(len(mmd_list_IVP)), mmd_list_IVP, label='MMD', marker='o')
        plt.plot(range(len(wassersten_list_IVP)), wassersten_list_IVP, label='Wasserstein', marker='s')
        plt.plot(range(len(weighted_wasserstein_list_IVP)), weighted_wasserstein_list_IVP, label='Weighted Wasserstein', marker='*')
        plt.plot(range(len(energy_list_IVP)), energy_list_IVP, label='Energy Distance', marker='^')
        # plt.plot(range(len(r2_list_IVP)), r2_list_IVP, label='R2 Score', marker='D')
        plt.xlabel("Timepoint t")
        plt.ylabel("Error")
        plt.title("Distribution Matching Error")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{fig_folder_address}/IVP_error.png')
        plt.close()

        metric_dict_IVP = {'mmd': mmd_list_IVP, 'wasserstein': wassersten_list_IVP, 'energy': energy_list_IVP, 'r2': r2_list_IVP, 'weighted_wasserstein': weighted_wasserstein_list_IVP}
        with open(f'{fig_folder_address}/IVP_error.json', 'w') as f:
            json.dump(metric_dict_IVP, f)