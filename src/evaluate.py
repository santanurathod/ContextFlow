import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance, energy_distance, wasserstein_distance_nd
import ot
import dcor

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


def compute_wasserstein_nd(X, Y):
    """
    sliced Wasserstein distance
    """
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    return ot.sliced.sliced_wasserstein_distance(X, Y, n_projections=200, p=2)

# def compute_wasserstein(X, Y):
#     # flatten to 1D for each feature or mean across features
#     X = X.cpu().numpy()
#     Y = Y.cpu().numpy()
#     return wasserstein_distance_nd(X, Y)

# def compute_energy(X, Y):
#     X_flat = X.cpu().numpy().flatten()
#     Y_flat = Y.cpu().numpy().flatten()
#     return energy_distance(X_flat, Y_flat)

def compute_energy(X, Y):
    """
    true multi-dimensional energy distance
    """
    X_flat = X.cpu().numpy()
    Y_flat = Y.cpu().numpy()
    return dcor.energy_distance(X_flat, Y_flat)

def compute_r2(X, Y):
    X_flat = X.cpu().numpy().flatten()
    Y_flat = Y.cpu().numpy().flatten()
    return r2_score(Y_flat, X_flat)

# def compute_weighted_wasserstein(X, Y, clf_model, gt_labels, label_encoder):
#     """
#     X: predicted
#     Y: true
#     clf_model: classifier model
#     gt_labels: true labels
#     label_encoder: label encoder
#     Returns:
#     weighted Wasserstein distance
#     """
#     X_flat = X.cpu().numpy().flatten()
#     Y_flat = Y.cpu().numpy().flatten()
    
#     pred_labels = clf_model.predict(X)
#     true_labels = label_encoder.transform(gt_labels)

#     # Assume X_flat, Y_flat, true_labels are numpy arrays of shape [N]
#     classes, counts = np.unique(true_labels, return_counts=True)
#     fractions = counts / counts.sum()

#     weighted_wass = 0.0

#     for cls, frac in zip(classes, fractions):
        
#             idx = np.where(true_labels == cls)[0]
#             x_cls = X_flat[idx]
#             y_cls = Y_flat[idx]
        
#             wass = wasserstein_distance(x_cls, y_cls)
#             weighted_wass += frac * wass

#     return weighted_wass

def compute_weighted_wasserstein(X, Y, clf_model, gt_labels, label_encoder):
    """
    X: predicted
    Y: true
    clf_model: classifier model
    gt_labels: true labels
    label_encoder: label encoder
    Returns:
    weighted Wasserstein distance
    """
    X= X.cpu().numpy()
    Y= Y.cpu().numpy()
    pred_labels = clf_model.predict(X)
    pred_label_names = label_encoder.inverse_transform(pred_labels)
    true_labels = label_encoder.transform(gt_labels)

    # Assume X_flat, Y_flat, true_labels are numpy arrays of shape [N]
    classes, counts = np.unique(true_labels, return_counts=True)
    fractions = counts / counts.sum()

    true_Z= 0

    weighted_wass = 0.0

    for cls, frac in zip(classes, fractions):
        
            idx_true = np.where(true_labels == cls)[0]
            idx_pred = np.where(pred_labels == cls)[0]

            if len(idx_true) == 0 or len(idx_pred) == 0:
                continue
        
            # x_cls = X[idx_pred].flatten()
            # y_cls = Y[idx_true].flatten()

            x_cls = X[idx_pred]
            y_cls = Y[idx_true]
            wass = ot.sliced.sliced_wasserstein_distance(x_cls, y_cls, n_projections=200, p=2)
            # wass = wasserstein_distance(x_cls, y_cls)
            weighted_wass += frac * wass
            true_Z+=frac
    
    weighted_wass = weighted_wass / true_Z
    return weighted_wass, pred_label_names

def evaluate_next_step(node, X_input, device, steps=400, n_points=1500, plot_distributions=False, weighted_wasserstein_data=None):
    mmd_list = []
    wasserstein_list = []
    energy_list = []
    r2_list = []
    weighted_wasserstein_list = []
    pred_celltype_names = []
    gt_celltype_names = []
    if weighted_wasserstein_data is not None:
        clf_model, gt_labels, label_encoder = weighted_wasserstein_data['clf_model'], weighted_wasserstein_data['gt_labels'], weighted_wasserstein_data['label_encoder']

    for t in range(len(X_input) - 1):
        print(f"Evaluating: time {t} → {t+1}")
        
        try:
            x0_np = X_input[t]
            x1_np = X_input[t + 1]
            n = min(n_points, min(len(x0_np), len(x1_np)))
            idx = np.random.choice(min(len(x0_np), len(x1_np)), size=n, replace=False)
            
            x0 = torch.tensor(x0_np[idx]).float().to(device)
            x1_true = torch.tensor(x1_np[idx]).float().to(device)
            t_span = torch.linspace(t, t+1, steps).to(device)
            gt_labels_sampled = [gt_labels[t+1][i] for i in idx]
            with torch.no_grad():
                traj = node.trajectory(x0, t_span)
            x1_pred = traj[-1]
            
            mmd_value = compute_mmd_multi_rbf(x1_pred.cpu(), x1_true.cpu())
            # wasserstein_value = compute_wasserstein(x1_pred, x1_true)
            wasserstein_value = compute_wasserstein_nd(x1_pred, x1_true)
            weighted_wasserstein_value, pred_label_names = compute_weighted_wasserstein(x1_pred, x1_true, clf_model, gt_labels_sampled, label_encoder)
            energy_value = compute_energy(x1_pred, x1_true)
            r2_value = compute_r2(x1_pred, x1_true)
            mmd_list.append(mmd_value)
            wasserstein_list.append(wasserstein_value)
            weighted_wasserstein_list.append(weighted_wasserstein_value)
            energy_list.append(energy_value)
            r2_list.append(r2_value)
            pred_celltype_names.append(pred_label_names)
            gt_celltype_names.append(gt_labels_sampled)
            print (f"t={t} → t+1: MMD={mmd_value:.4f}, Wasserstein={wasserstein_value:.4f}, "
                f"Energy={energy_value:.4f}, R2={r2_value:.4f}, Weighted Wasserstein={weighted_wasserstein_value:.4f}")
            if plot_distributions:
                plt.figure(figsize=(6,4))
                sns.kdeplot(x1_pred.cpu().numpy().flatten(), label="Predicted", fill=True)
                sns.kdeplot(x1_true.cpu().numpy().flatten(), label="True", fill=True)
                plt.title(f"Distribution at t={t+1}")
                plt.xlabel("Feature value (flattened)")
                plt.legend()
                plt.tight_layout()
                plt.show()
        except:
            import pdb; pdb.set_trace()

    return mmd_list, wasserstein_list, energy_list, r2_list, weighted_wasserstein_list, pred_celltype_names, gt_celltype_names


def evaluate_IVP(trajectory, X_input, device, steps=400, n_points=1500, plot_distributions=False, weighted_wasserstein_data=None):
    mmd_list = []
    wasserstein_list = []
    energy_list = []
    r2_list = []
    pred_celltype_names = []
    weighted_wasserstein_list = []
    gt_celltype_names = []
    if weighted_wasserstein_data is not None:
        clf_model, gt_labels, label_encoder = weighted_wasserstein_data['clf_model'], weighted_wasserstein_data['gt_labels'], weighted_wasserstein_data['label_encoder']

    n_times = len(X_input)
    traj_eval_times = [int(i*(steps-1)/(n_times-1)) for i in range(n_times)]
    for t in range(n_times):
        x1_pred = trajectory[traj_eval_times[t]]
        x1_true = torch.tensor(X_input[t]).float().to(device)
        if x1_pred.shape[1]> x1_true.shape[1]:
            x1_pred = x1_pred[:, :x1_true.shape[1]]
        mmd_value = compute_mmd_multi_rbf((x1_pred).cpu(), (x1_true).cpu())
        # wasserstein_value = compute_wasserstein(x1_pred, x1_true)
        wasserstein_value = compute_wasserstein_nd(x1_pred, x1_true)
        weighted_wasserstein_value, pred_label_names = compute_weighted_wasserstein(x1_pred, x1_true, clf_model, gt_labels[t], label_encoder)
        energy_value = compute_energy(x1_pred, x1_true)
        # r2_value = compute_r2(x1_pred, x1_true)
        mmd_list.append(mmd_value)
        wasserstein_list.append(wasserstein_value)
        weighted_wasserstein_list.append(weighted_wasserstein_value)
        energy_list.append(energy_value)
        pred_celltype_names.append(pred_label_names)
        gt_celltype_names.append(gt_labels[t])
        print (f"At t={t}: MMD={mmd_value:.4f}, Wasserstein={wasserstein_value:.4f}, "
               f"Energy={energy_value:.4f}, Weighted Wasserstein={weighted_wasserstein_value:.4f}")
        
    return mmd_list, wasserstein_list, energy_list, r2_list, weighted_wasserstein_list, pred_celltype_names, gt_celltype_names