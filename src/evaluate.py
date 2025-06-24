import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance, energy_distance


def compute_mmd_multi_rbf(X, Y, gammas=[2, 1, 0.5, 0.1, 0.01, 0.005]):
    """
    Compute multi-kernel MMD^2 between X and Y with multiple RBF gammas.
    X: [N, D]
    Y: [M, D]
    gammas: list of gamma values (1 / (2*sigma^2))
    Returns:
    mean MMD^2 across gammas
    """
    XX = torch.cdist(X, X, p=2)**2
    YY = torch.cdist(Y, Y, p=2)**2
    XY = torch.cdist(X, Y, p=2)**2
    m = X.size(0)
    n = Y.size(0)
    mmd_total = 0.0
    for gamma in gammas:
        K_XX = torch.exp(-gamma * XX)
        K_YY = torch.exp(-gamma * YY)
        K_XY = torch.exp(-gamma * XY)
        mmd2 = (K_XX.sum() - torch.diagonal(K_XX).sum()) / (m * (m - 1)) \
             + (K_YY.sum() - torch.diagonal(K_YY).sum()) / (n * (n - 1)) \
             - 2 * K_XY.mean()
        mmd_total += mmd2
    mean_mmd = mmd_total / len(gammas)
    return mean_mmd.item()

def compute_wasserstein(X, Y):
    # flatten to 1D for each feature or mean across features
    X_flat = X.cpu().numpy().flatten()
    Y_flat = Y.cpu().numpy().flatten()
    return wasserstein_distance(X_flat, Y_flat)

def compute_energy(X, Y):
    X_flat = X.cpu().numpy().flatten()
    Y_flat = Y.cpu().numpy().flatten()
    return energy_distance(X_flat, Y_flat)

def compute_r2(X, Y):
    X_flat = X.cpu().numpy().flatten()
    Y_flat = Y.cpu().numpy().flatten()
    return r2_score(Y_flat, X_flat)


def evaluate_next_step(node, X_input, device, steps=400, n_points=1500, plot_distributions=False):
    mmd_list = []
    wasserstein_list = []
    energy_list = []
    r2_list = []

    for t in range(len(X_input) - 1):
        print(f"Evaluating: time {t} → {t+1}")
        
        x0_np = X_input[t]
        x1_np = X_input[t + 1]
        n = min(n_points, min(len(x0_np), len(x1_np)))
        idx = np.random.choice(min(len(x0_np), len(x1_np)), size=n, replace=False)
        
        x0 = torch.tensor(x0_np[idx]).float().to(device)
        x1_true = torch.tensor(x1_np[idx]).float().to(device)
        t_span = torch.linspace(t, t+1, steps).to(device)
        with torch.no_grad():
            traj = node.trajectory(x0, t_span)
        x1_pred = traj[-1]
        mmd_value = compute_mmd_multi_rbf(x1_pred.cpu(), x1_true.cpu())
        wasserstein_value = compute_wasserstein(x1_pred, x1_true)
        energy_value = compute_energy(x1_pred, x1_true)
        r2_value = compute_r2(x1_pred, x1_true)
        mmd_list.append(mmd_value)
        wasserstein_list.append(wasserstein_value)
        energy_list.append(energy_value)
        r2_list.append(r2_value)
        print (f"t={t} → t+1: MMD={mmd_value:.4f}, Wasserstein={wasserstein_value:.4f}, "
               f"Energy={energy_value:.4f}, R2={r2_value:.4f}")
        if plot_distributions:
            plt.figure(figsize=(6,4))
            sns.kdeplot(x1_pred.cpu().numpy().flatten(), label="Predicted", fill=True)
            sns.kdeplot(x1_true.cpu().numpy().flatten(), label="True", fill=True)
            plt.title(f"Distribution at t={t+1}")
            plt.xlabel("Feature value (flattened)")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return mmd_list, wasserstein_list, energy_list, r2_list


def evaluate_IVP(trajectory, X_input, device, steps=400, n_points=1500, plot_distributions=False):
    mmd_list = []
    wasserstein_list = []
    energy_list = []
    r2_list = []

    n_times = len(X_input)
    traj_eval_times = [int(i*(steps/(n_times-1))-1) for i in range(n_times)]

    for t in range(n_times):
        x1_pred = trajectory[traj_eval_times[t]]
        x1_true = torch.tensor(X_input[t]).float().to(device)
        if x1_pred.shape[1]> x1_true.shape[1]:
            x1_pred = x1_pred[:, :x1_true.shape[1]]
        mmd_value = compute_mmd_multi_rbf((x1_pred).cpu(), (x1_true).cpu())
        wasserstein_value = compute_wasserstein(x1_pred, x1_true)
        energy_value = compute_energy(x1_pred, x1_true)
        # r2_value = compute_r2(x1_pred, x1_true)
        mmd_list.append(mmd_value)
        wasserstein_list.append(wasserstein_value)
        energy_list.append(energy_value)
        r2_list.append(0)
        print (f"At t={t}: MMD={mmd_value:.4f}, Wasserstein={wasserstein_value:.4f}, "
               f"Energy={energy_value:.4f}")
        
    return mmd_list, wasserstein_list, energy_list, r2_list