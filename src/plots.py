import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import scprep



from torchdyn.core import NeuralODE


def plot_trajectories(scRNA, traj, day_list, fig_address, legend=True):
    n = 200
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scprep.plot.scatter(
        scRNA.obsm["X_umap"][:, 0],
        scRNA.obsm["X_umap"][:, 1],
        c=day_list,
        ax=ax,
    )
    # ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    ax.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.4, alpha=0.1, c="olive")
    # ax.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")

    for i in range(10):
        ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.9, c="black")
    if legend:
        plt.legend([r"$p_0$", r"$p_t$", r"$p_1$", r"$X_t \mid X_0$"])
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis("off")

    plt.savefig(fig_address)


def plot_trajectories_new(traj,df, labels_dict, legend=True):
    n = 20
    X_here= df.values
    labels= [labels_dict[idx.split('_')[-1]] for idx in df.index]
    # import pdb; pdb.set_trace()
    for ng in range(4):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # scprep.plot.scatter(
        #     X_here[:, ng],
        #     X_here[:, ng+1],
        #     c=labels,
        #     ax=ax,
        # )
        scatter = ax.scatter(
        X_here[:, ng],  # Replace with the desired data for the x-axis
        X_here[:, ng+1],  # Replace with the desired data for the y-axis
        s=1,
        c=labels,
        cmap='viridis' # Optional: choose a colormap
    )

        plt.legend(handles=scatter.legend_elements()[0], labels=labels_dict.keys())

        # # ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
        ax.scatter(traj[:, :n, ng], traj[:, :n, ng+1], s=10, alpha=0.1, c="olive")
        # # ax.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")

        for i in range(15):
            ax.plot(traj[:, i, ng], traj[:, i, ng+1], alpha=0.9, c="black")
        # # if legend:
        #     # plt.legend([r"$p_0$", r"$p_t$", r"$p_1$", r"$X_t \mid X_0$"])
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.axis("off")
