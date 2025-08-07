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

from src.ode_helper_functions import *
from src.data_loading import preprocess_data, process_data, get_batch
from src.plots import plot_trajectories, plot_trajectories_new
from src.evaluate import *

from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP, GradModel, Graph_like_transformer
from torchcfm.utils import torch_wrapper

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
        dim=(params['dim']+1 if params['use_celltype'] else params['dim']),
        out_dim=params['out_dim'],
        time_varying=params['time_varying'],
        w=params['w']
    ).to(device)

    ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), params['learning_rate'])
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=params['sigma'])

    losses = []
    for i in tqdm(range(params['n_epochs'])):
        ot_cfm_optimizer.zero_grad()
        t, xt, ut, xt_conditional = get_batch(FM, X_phate, X_phate_conditional, batch_size, n_times, lambda_=params['lambda_'], lambda_bio_prior=params['lambda_bio_prior'], Spatial=Spatial, Celltype_list=Celltype_list, device=device)
        if params['use_celltype']:
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
    args = parser.parse_args()

    fig_folder_address = f'./experiment_figures/{args.train_config}'
    os.makedirs(fig_folder_address, exist_ok=True)
    
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
    X_phate, X_phate_conditional, Spatial, Celltype_list = process_data(scRNA, X_raw, cell_type_key, labels_dict, params['use_spatial'], params['use_celltype'], params['use_bio_prior'], params['representation'])

    n_times = len(X_phate)

    ot_cfm_model = train(params, n_times, X_phate, X_phate_conditional, device, fig_folder_address, Spatial, Celltype_list)

    if params['use_celltype']:
        cond_dim = 1
    else:
        cond_dim = 0
    node = NeuralODE(ConditionalODEVectorField(ot_cfm_model, cond_dim, params['dim']), solver="dopri5", sensitivity="adjoint")
    with torch.no_grad():
        if params['use_celltype']:
            cfm_input = torch.cat([
                torch.from_numpy(X_phate[0][:1000]).float().to(device),
                torch.from_numpy(X_phate_conditional[0][:1000][:, None]).float().to(device)
            ], dim=-1)
        else:
            cfm_input = torch.from_numpy(X_phate[0][:1000]).float().to(device)
               
        traj = node.trajectory(
            cfm_input,
            t_span=torch.linspace(0, n_times - 1, params['trajectory_steps']),
        ).cpu()

        traj_fig_address = f'{fig_folder_address}/trajectory.png'
        plot_trajectories(scRNA, traj.cpu().numpy(), day_list, traj_fig_address)
    
    try:
        mmd_list_next_step, wassersten_list_next_step, energy_list_next_step, r2_list_next_step = evaluate_next_step(node, X_phate, device)
        plt.figure(figsize=(8,6))
        plt.plot([1+i for i in range(len(mmd_list_next_step))], mmd_list_next_step, label='MMD', marker='o')
        plt.plot([1+i for i in range(len(wassersten_list_next_step))], wassersten_list_next_step, label='Wasserstein', marker='s')
        plt.plot([1+i for i in range(len(energy_list_next_step))], energy_list_next_step, label='Energy Distance', marker='^')
        plt.plot([1+i for i in range(len(r2_list_next_step))], r2_list_next_step, label='R2 Score', marker='D')
        plt.xlabel("Timepoint t → t+1")
        plt.ylabel("Error")
        plt.title("Distribution Matching Error")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{fig_folder_address}/next_step_error.png')
        plt.close()

        metric_dict = {'mmd': mmd_list_next_step, 'wasserstein': wassersten_list_next_step, 'energy': energy_list_next_step, 'r2': r2_list_next_step}
        with open(f'{fig_folder_address}/next_step_error.json', 'w') as f:
            json.dump(metric_dict, f)
    except:
        print("Error in evaluating next step")

    mmd_list_IVP, wassersten_list_IVP, energy_list_IVP, r2_list_IVP = evaluate_IVP(traj, X_phate, device)

    plt.figure(figsize=(8,6))
    plt.plot(range(len(mmd_list_IVP)), mmd_list_IVP, label='MMD', marker='o')
    plt.plot(range(len(wassersten_list_IVP)), wassersten_list_IVP, label='Wasserstein', marker='s')
    plt.plot(range(len(energy_list_IVP)), energy_list_IVP, label='Energy Distance', marker='^')
    # plt.plot(range(len(r2_list_IVP)), r2_list_IVP, label='R2 Score', marker='D')
    plt.xlabel("Timepoint t")
    plt.ylabel("Error")
    plt.title("Distribution Matching Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{fig_folder_address}/IVP_error.png')
    plt.close()

    metric_dict_IVP = {'mmd': mmd_list_IVP, 'wasserstein': wassersten_list_IVP, 'energy': energy_list_IVP, 'r2': r2_list_IVP}
    with open(f'{fig_folder_address}/IVP_error.json', 'w') as f:
        json.dump(metric_dict_IVP, f)