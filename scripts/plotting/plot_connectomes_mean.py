import os
import pandas as pd
import seaborn as sns
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from nilearn.plotting import plot_connectome, view_connectome
import matplotlib.pyplot as plt
from nilearn import datasets
from tqdm import tqdm
from sklearn import preprocessing

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot import get_lower_tri_heatmap


### mean functional connectivity plots
results_root = (
    "/storage/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
)
plots_root = (
    "/storage/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs"
)
n_parcels = 400
trim_length = None

dir_name = (
    f"connectomes_nparcels-{n_parcels}_tasktype-natural_trim-{trim_length}.pkl"
)
results_pkl = os.path.join(results_root, dir_name)
fc_data = pd.read_pickle(results_pkl)

mats_dir = os.path.join(
    plots_root, f"mean_connectivity_matrices_nparcels-{n_parcels}"
)
output_dir = os.path.join(plots_root, mats_dir)
os.makedirs(output_dir, exist_ok=True)

coords_file = f"Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"

atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir="/storage/store3/work/haggarwa/connectivity",
    resolution_mm=2,
    n_rois=n_parcels,
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
network_labels = []
rename_networks = {
    "Vis": "Visual",
    "Cont": "FrontPar",
    "SalVentAttn": "VentAttn",
}
for network in networks:
    components = network.split("_")
    components[2] = rename_networks.get(components[2], components[2])
    hemi_network = " ".join(components[1:3])
    hemi_network_labels.append(hemi_network)
    network_labels.append(components[2])
ticks = []
unique_hemi_network_labels = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
        unique_hemi_network_labels.append(label)

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(network_labels)
unique_encoded_labels = np.unique(encoded_labels)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]

sns.set_context("notebook", font_scale=1.05)
coords = pd.read_csv(
    os.path.join(
        "/storage/store3/work/haggarwa/connectivity",
        coords_file,
    )
)[["R", "A", "S"]].to_numpy()
for cov in cov_estimators:
    for measure in measures:
        try:
            matrix = vec_to_sym_matrix(
                np.nanmean(
                    np.vstack(list(fc_data[f"{cov} {measure}"])),
                    axis=0,
                ),
                diagonal=np.ones(n_parcels),
            )
        except ValueError as e:
            print(e)
            print(f"{sub} {task} {cov} {measure} does not exist")
            continue
        sns.set_context("notebook")
        get_lower_tri_heatmap(
            matrix,
            title=f"{cov} {measure}",
            output=os.path.join(mats_dir, f"full_{cov}_{measure}"),
            ticks=ticks,
            labels=unique_hemi_network_labels,
            grid=True,
            diag=True,
            triu=True,
        )

        # get network wise average connectivity
        network_pair_conns = np.zeros(
            (len(unique_encoded_labels), len(unique_encoded_labels))
        )
        for network_i in unique_encoded_labels:
            index_i = np.where(encoded_labels == network_i)[0]
            for network_j in unique_encoded_labels:
                index_j = np.where(encoded_labels == network_j)[0]
                matrix[np.triu_indices_from(matrix)] = np.nan
                network_pair_conn = np.nanmean(
                    matrix[np.ix_(index_i, index_j)]
                )
                network_pair_conns[network_i][network_j] = network_pair_conn
        # plot network wise average connectivity
        get_lower_tri_heatmap(
            network_pair_conns,
            figsize=(5, 5),
            title=f"{cov} {measure}",
            output=os.path.join(mats_dir, f"networks_{cov}_{measure}"),
            labels=le.inverse_transform(unique_encoded_labels),
            triu=True,
            cmap="viridis",
        )
        plt.close("all")
