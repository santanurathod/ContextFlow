import math
import warnings
from functools import partial
from typing import Optional, Union

import numpy as np
# import ot as pot
import ot_modified as pot
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd


import torch
import torch.nn.functional as F

class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        """Initialize the OTPlanSampler class.

        Parameters
        ----------
        method: str
            choose which optimal transport solver you would like to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced",
            "partial"] OT solvers.
        reg: float, optional
            regularization parameter to use for Sinkhorn-based iterative solvers.
        reg_m: float, optional
            regularization weight for unbalanced Sinkhorn-knopp solver.
        normalize_cost: bool, optional
            normalizes the cost matrix so that the maximum cost is 1. Helps
            stabilize Sinkhorn-based solvers. Should not be used in the vast
            majority of cases.
        num_threads: int or str, optional
            number of threads to use for the "exact" OT solver. If "max", uses
            the maximum number of threads.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        """
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "sinkhorn_relative_entropy":
            self.ot_fn = partial(pot.sinkhorn, reg=reg, method="sinkhorn_relative_entropy")
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn
        self.method = method

    def get_biological_map_prior(self, ct0, ct1):
        
        # the file name is hardcoded for now, need to make it more general
        feasible_matrix= pd.read_csv('/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_type_feasibility_matrix_GSE232025.csv', index_col=0)
        submatrix = feasible_matrix.loc[ct0, ct1]
        bio_prior = np.where(submatrix.values, 0, -100000)
        return bio_prior

    def get_communication_matrix(self, ct0, ct1, cc_communication_type, cc_index):

        if cc_communication_type == 'all_at_once':
            communication_matrix= pd.read_csv('/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_cell_communication_GSE232025/all_at_once_GSE232025.csv', index_col=0)
            
            submatrix = communication_matrix.loc[ct0, ct1]
            Q_prior = submatrix.values
        elif cc_communication_type == 'step_by_step':
            communication_matrix= pd.read_csv(f'/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/metadata/cell_cell_communication_GSE232025/step_by_step_{cc_index}_GSE232025.csv', index_col=0)
            
            submatrix = communication_matrix.loc[ct0, ct1]
            Q_prior = submatrix.values
        else:
            raise ValueError(f"Unknown communication type: {cc_communication_type}")

        Q_prior = Q_prior / Q_prior.max()

        return Q_prior
        
        
    def get_relative_entropy_prior(self, p0=None, p1=None, ct0=None, ct1=None, ot_reg_variation= 'relativeentropic_g+p', cc_communication_type=None, cc_index=None, ot_reg_lambda=None):
        
        if 'p' in ot_reg_variation.split('_')[1]:

            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
 
            P_Dist= torch.cdist(p0, p1)**2

            ## kind of normalization to try
            #1, reverts to uniform plan but surprisingly works better
            # P_Dist_norm = scaler.fit_transform(P_Dist)  # Now in [0, 1]
            # P_Dist_norm = torch.from_numpy(P_Dist_norm)
            # P_Dist= P_Dist_norm

            P_Dist= P_Dist / P_Dist.max()

            
        # import pdb; pdb.set_trace()

        if ot_reg_variation == 'relativeentropic_g+p':
            Q_prior=F.softmax(-P_Dist, dim=1).detach().cpu().numpy()

            
        elif ot_reg_variation == 'relativeentropic_g+p+c' and cc_communication_type == 'step_by_step': # when we consider communication between timepoints
            assert cc_index!=None
            cc_index= int(cc_index)
            communication_matrix= self.get_communication_matrix(ct0, ct1, 'step_by_step', cc_index)
            communication_matrix= communication_matrix / communication_matrix.max()
            P_Dist= ot_reg_lambda*P_Dist+(1-ot_reg_lambda)*communication_matrix
            Q_prior=F.softmax(-P_Dist, dim=1).detach().cpu().numpy()

        elif ot_reg_variation == 'relativeentropic_g+p+c' and cc_communication_type == 'all_at_once': # when we consider overall communication between timepoints
            communication_matrix= self.get_communication_matrix(ct0, ct1, 'all_at_once', cc_index)
            communication_matrix= communication_matrix / communication_matrix.max()
            P_Dist= ot_reg_lambda*P_Dist+(1-ot_reg_lambda)*communication_matrix
            Q_prior=F.softmax(-P_Dist, dim=1).detach().cpu().numpy()

        else:
            raise ValueError(f"Unknown variation kind: {variation_kind}")
        
        return Q_prior
        

        
    def get_map(self, x0, x1, p0=None, p1=None, ct0=None, ct1=None, method="exact", cc_index=None, params=None):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """

        lambda_= params['lambda_']
        lambda_bio_prior= params['lambda_bio_prior']
        cc_communication_type= params['cc_communication_type']
        ot_cost_variation= params['OT_cost_variation']
        ot_reg_variation= params['OT_reg_variation']
        ot_reg_lambda= params['OT_reg_lambda']


        if ot_reg_variation == 'relativeentropic_g+p+c':
            assert cc_communication_type!=None


        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)

        gene_dist = torch.cdist(x0, x1)**2
        
        # # the normalization is something I'm trying, earlier it wasn't the case so let's see if it works
        # gene_dist = gene_dist / gene_dist.max()

        # import pdb; pdb.set_trace()
        # if p0 is not None and p1 is not None and ct0 is not None and ct1 is not None:
        if ot_cost_variation == 'g+p+f':
            pair_dist = torch.cdist(p0, p1)**2
            pair_dist = pair_dist / pair_dist.max()
            M = lambda_*gene_dist + (1-lambda_)*pair_dist
            bio_prior = self.get_biological_map_prior(ct0, ct1)
            M = M + lambda_bio_prior*bio_prior
        # elif p0 is not None and p1 is not None and ct0 is None and ct1 is None:
        elif ot_cost_variation == 'g+p':
            pair_dist = torch.cdist(p0, p1)**2
            pair_dist = pair_dist / pair_dist.max()
            M = lambda_*gene_dist + (1-lambda_)*pair_dist
        else:
            M = gene_dist
        
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        
        # import pdb; pdb.set_trace()

        if method == "sinkhorn_relative_entropy" and ot_reg_variation == 'relativeentropic_g+p': # working only spatial variation for now
            Q_prior = self.get_relative_entropy_prior(p0=p0, p1=p1, ot_reg_variation=ot_reg_variation)
            p = self.ot_fn(a=a, b=b, M=M.detach().cpu().numpy(), Q_prior=Q_prior)
        elif method == "sinkhorn_relative_entropy" and ot_reg_variation == 'relativeentropic_g+p+c':
            Q_prior = self.get_relative_entropy_prior(p0=p0, p1=p1, ct0=ct0, ct1=ct1, ot_reg_variation=ot_reg_variation, cc_communication_type=cc_communication_type, cc_index=cc_index, ot_reg_lambda=ot_reg_lambda)
            p = self.ot_fn(a=a, b=b, M=M.detach().cpu().numpy(), Q_prior=Q_prior)
        else:
            p = self.ot_fn(a=a, b=b, M=M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size

        return p

    def sample_map(self, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, p0=None, p1=None, ct0=None, ct1=None, replace=True, method="exact", cc_index=None, params=None):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        
        pi = self.get_map(x0, x1, p0, p1, ct0, ct1, method=method, cc_index=cc_index, params=params)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target labeled samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs)
            represents the source label minibatch
        y1 : Tensor, shape (bs)
            represents the target label minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the target minibatch drawn from $\pi$
        y0[i] : Tensor, shape (bs, *dim)
            represents the source label minibatch drawn from $\pi$
        y1[j] : Tensor, shape (bs, *dim)
            represents the target label minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, X):
        """Compute the OT trajectories between different sample populations moving from the source
        to the target distribution.

        Parameters
        ----------
        X : Tensor, (bs, times, *dim)
            different populations of samples moving from the source to the target distribution.

        Returns
        -------
        to_return : Tensor, (bs, times, *dim)
            represents the OT sampled trajectories over time.
        """
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret
