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
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot import get_lower_tri_heatmap


### functional connectivity plots
results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
plots_root = (
    "/data/parietal/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs"
)
n_parcels = 400
trim_length = None

dir_name = f"classification-across_tasktype-natural_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")
fc_data = pd.read_pickle(results_pkl)

mats_dir = os.path.join(
    plots_root, f"connectivity_matrices_nparcels-{n_parcels}"
)
brain_dir = os.path.join(
    plots_root, f"brain_connectivity_nparcels-{n_parcels}"
)
html_dir = os.path.join(
    plots_root, f"brain_connectivity_html_nparcels-{n_parcels}"
)
for directory in [mats_dir, brain_dir, html_dir]:
    output_dir = os.path.join(plots_root, dir_name)
    os.makedirs(output_dir, exist_ok=True)

coords_file = f"Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"

# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
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
os.makedirs(mats_dir, exist_ok=True)
os.makedirs(brain_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)
coords = pd.read_csv(
    os.path.join(
        cache,
        "schaefer_2018",
        coords_file,
    )
)[["R", "A", "S"]].to_numpy()
for sub in tqdm(np.unique(fc_data["subject_ids"])):
    for task in tasks:
        for cov in cov_estimators:
            for measure in measures:
                try:
                    matrix = vec_to_sym_matrix(
                        np.mean(
                            np.vstack(
                                list(
                                    fc_data[
                                        (fc_data["subject_ids"] == sub)
                                        & (fc_data["tasks"] == task)
                                    ][f"{cov} {measure}"]
                                )
                            ),
                            axis=0,
                        ),
                        diagonal=np.ones(n_parcels),
                    )
                except ValueError as e:
                    print(e)
                    print(f"{sub} {task} {cov} {measure} does not exist")
                    continue
                get_lower_tri_heatmap(
                    matrix,
                    title=f"{sub} {task}",
                    output=os.path.join(
                        mats_dir, f"{sub}_{task}_{cov}_{measure}"
                    ),
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
                        network_pair_conns[network_i][
                            network_j
                        ] = network_pair_conn
                # plot network wise average connectivity
                get_lower_tri_heatmap(
                    network_pair_conns,
                    title=f"{sub} {task}",
                    output=os.path.join(
                        mats_dir, f"{sub}_{task}_{cov}_{measure}_networks"
                    ),
                    labels=le.inverse_transform(unique_encoded_labels),
                    triu=True,
                    cmap="hot_r",
                )

                # plot connectome on glass brain
                f = plt.figure(figsize=(9, 4))
                plot_connectome(
                    matrix,
                    coords,
                    edge_threshold="99.8%",
                    title=f"{sub} {task}",
                    node_size=25,
                    figure=f,
                    colorbar=True,
                    output_file=os.path.join(
                        brain_dir,
                        f"{sub}_{task}_{cov}_{measure}_connectome.png",
                    ),
                )
                threshold = np.percentile(matrix, 99.8)
                matrix_thresholded = np.where(matrix > threshold, matrix, 0)
                max_ = np.max(matrix)

                # plot connectome in 3D view in html
                three_d = view_connectome(
                    matrix,
                    coords,
                    # edge_threshold="99.8%",
                    symmetric_cmap=False,
                    title=f"{sub} {task}",
                )
                three_d.save_as_html(
                    os.path.join(
                        html_dir,
                        f"{sub}_{task}_{cov}_{measure}_connectome.html",
                    )
                )
                plt.close("all")
