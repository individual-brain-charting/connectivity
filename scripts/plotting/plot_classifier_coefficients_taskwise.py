from utils.plot import (
    _load_transform_weights,
    get_lower_tri_heatmap,
    get_network_labels,
)
import os
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from sklearn import preprocessing
from nilearn import datasets
import matplotlib.pyplot as plt
from matplotlib import cm

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
plots_root = (
    "/data/parietal/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs"
)
n_parcels = 400
trim_length = None

dir_name = f"weights_taskwise_nparcels-{n_parcels}_trim-{trim_length}"
weight_dir = os.path.join(results_root, dir_name)

output_dir = os.path.join(plots_root, dir_name)
os.makedirs(output_dir, exist_ok=True)

cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
measures = ["correlation", "partial correlation"]
classify = ["Tasks", "Subjects", "Runs"]

# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir="/data/parietal/store3/work/haggarwa/connectivity",
    resolution_mm=2,
    n_rois=n_parcels,
)

for classify_by in classify:
    for cov_estimator in cov_estimators:
        for measure in measures:
            weights = np.load(
                os.path.join(
                    weight_dir,
                    f"{classify_by}_{cov_estimator} {measure}_weights.npy",
                )
            )
            # take the absolute value of the weights
            weights = np.abs(weights)
            task_labels = np.load(
                os.path.join(
                    weight_dir,
                    f"{classify_by}_{cov_estimator} {measure}_labels.npy",
                ),
                allow_pickle=True,
            )
            network_labels = get_network_labels(atlas)[1]
            le = preprocessing.LabelEncoder()
            encoded_labels = le.fit_transform(network_labels)
            unique_labels = np.unique(encoded_labels)

            fig, axs = plt.subplots(2, 3, figsize=(8, 4))
            for i in range(weights.shape[0]):
                ax = axs[i // 3, i % 3]
                task = task_labels[i]
                weights_mat = vec_to_sym_matrix(
                    weights[i, :], diagonal=np.ones(n_parcels)
                )
                network_pair_weights = np.zeros(
                    (len(unique_labels), len(unique_labels))
                )
                # get network pair weights
                for network_i in unique_labels:
                    index_i = np.where(encoded_labels == network_i)[0]
                    for network_j in unique_labels:
                        index_j = np.where(encoded_labels == network_j)[0]
                        weights_mat[np.triu_indices_from(weights_mat)] = np.nan
                        network_pair_weight = np.nanmean(
                            weights_mat[np.ix_(index_i, index_j)]
                        )
                        network_pair_weights[network_i][
                            network_j
                        ] = network_pair_weight
                # mask upper triangle
                mask = np.triu(
                    np.ones_like(network_pair_weights, dtype=bool), 1
                )
                network_pair_weights[mask] = np.nan
                # plot network pair weights
                im = ax.imshow(
                    network_pair_weights,
                    cmap="viridis",
                )
                ax.set_xticks(np.arange(len(unique_labels)))
                ax.set_yticks(np.arange(len(unique_labels)))
                ax.set_xticklabels(
                    le.inverse_transform(unique_labels),
                    rotation=40,
                    ha="right",
                )
                ax.set_yticklabels(
                    le.inverse_transform(unique_labels),
                    rotation=0,
                )
                ax.set_title(f"{task}")
                cmap = plt.get_cmap()
                cmap.set_bad("white")
            for ax in axs.flat:
                ax.label_outer(remove_inner_ticks=True)
            fig.suptitle(f"{cov_estimator} {measure}")
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"{classify_by}_{cov_estimator}_{measure}.png",
                ),
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"{classify_by}_{cov_estimator}_{measure}.svg",
                ),
                bbox_inches="tight",
                transparent=True,
            )
            plt.close(fig)
