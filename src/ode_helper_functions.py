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

import os

import torch
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP, GradModel, Graph_like_transformer
from torchcfm.utils import plot_trajectories, torch_wrapper
import pandas as pd


import torch.nn as nn
import torch.nn.functional as F

class ConditionalODEVectorField(nn.Module):
    def __init__(self, model, cond_dim, evolve_dim):
        super().__init__()
        self.model = model
        self.cond_dim = cond_dim
        self.evolve_dim = evolve_dim
        self.total_dim = cond_dim + evolve_dim


    # --- Add 'args=None' to the signature ---
    def forward(self, t, x, args=None):
        # x is the full state [gene_conditional, metabolic_input], shape [B, total_dim]
        # t is the current time
        # args is an optional argument passed by torchdyn (we can ignore it if unused)



        # 1. Prepare input for the underlying model
        if x.dim() > 1: # Handle batch dimension
             t_expanded = t.expand(x.shape[0], 1) if isinstance(t, torch.Tensor) else torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        else: # Handle single sample
             t_expanded = t.reshape(1, 1) if isinstance(t, torch.Tensor) else torch.tensor([[t]], device=x.device, dtype=x.dtype)

        model_input = torch.cat([x, t_expanded], dim=-1)

        # 2. Get the derivative for the evolving part
        d_evolve_dt = self.model(model_input)

        # 3. Create the zero derivative for the conditional part
        zeros_for_cond = torch.zeros_like(x[:, self.evolve_dim:self.total_dim])

        # 4. Concatenate
        full_derivative = torch.cat([d_evolve_dt, zeros_for_cond], dim=-1)

        # 5. Assert shapes
        assert full_derivative.shape == x.shape, \
            f"Output derivative shape {full_derivative.shape} must match input state shape {x.shape}"

        return full_derivative
    

class TorchWrapperWithConditional(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, args=None):
        # x is expected to have shape [batch_size, N+1]
        return self.model(x, t)