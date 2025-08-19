### fc-sc similarity, network-wise matrix

import os
import pandas as pd
import seaborn as sns
import numpy as np
from nilearn.connectome import vec_to_sym_matrix, sym_matrix_to_vec
from nilearn import datasets
from sklearn import preprocessing
import sys
from matplotlib import pyplot as plt

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.plot import (
    get_network_labels,
    get_lower_tri_heatmap,
    mean_connectivity,
)

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

# root = "/data/parietal/store3/work/haggarwa/connectivity"
root = "/Users/himanshu/Desktop/ibc/connectivity"
results_root = os.path.join(
    root,
    "results",
)
plots_root = os.path.join(root, "plots", "wo_extra_GBU_runs")


n_parcels = 400
trim_length = None
do_hatch = False
labels_fmt = "network"  # "hemi network" or "network"

fc_data_path = os.path.join(
    results_root,
    "wo_extra_GBU_runs",
    f"connectomes_nparcels-{n_parcels}_tasktype-natural_trim-{trim_length}.pkl",
)
sc_data_path = os.path.join(
    results_root,
    f"sc_data_native_{n_parcels}",
)
fc_data = pd.read_pickle(fc_data_path)
fc_data = fc_data[fc_data["dataset"] == "ibc"]
sc_data = pd.read_pickle(sc_data_path)

out_dir_name = (
    f"fcsc_similarity_networkwise_nparcels-{n_parcels}_trim-{trim_length}"
)
output_dir = os.path.join(plots_root, out_dir_name)
os.makedirs(output_dir, exist_ok=True)


# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
    "LePetitPrince",
]

# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=root, resolution_mm=2, n_rois=n_parcels
)

fc_data = mean_connectivity(fc_data, tasks, cov_estimators, measures)
fc_data.reset_index(drop=True, inplace=True)
sc_data.reset_index(drop=True, inplace=True)

if labels_fmt == "hemi network":
    labels = get_network_labels(atlas)[0]
elif labels_fmt == "network":
    labels = get_network_labels(atlas)[1]

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)

unique_labels = np.unique(encoded_labels)

results = []
for cov in cov_estimators:
    for measure in measures:
        for task in tasks:
            func = fc_data[fc_data["task"] == task]
            func = func[func["measure"] == cov + " " + measure]
            # get func and struc conn for each subject
            for sub in func["subject"].unique():
                sub_func = func[func["subject"] == sub]
                sub_func_mat = vec_to_sym_matrix(
                    sub_func["connectivity"].values[0],
                    diagonal=np.ones(n_parcels),
                )
                sub_func_mat[np.triu_indices_from(sub_func_mat)] = np.nan
                sub_struc = sc_data[sc_data["subject"] == sub]
                sub_struc_mat = vec_to_sym_matrix(
                    sub_struc["connectivity"].values[0],
                    diagonal=np.ones(n_parcels),
                )
                sub_struc_mat[np.triu_indices_from(sub_struc_mat)] = np.nan

                # create empty matrix for network pair correlations
                network_pair_corr = np.zeros(
                    (len(unique_labels), len(unique_labels))
                )
                print(f"\n\n{task} {sub} {cov} {measure}\n\n")
                # get the nodes indices for each network
                for network_i in unique_labels:
                    index_i = np.where(encoded_labels == network_i)[0]
                    # print(index_i)
                    for network_j in unique_labels:
                        index_j = np.where(encoded_labels == network_j)[0]
                        # print(index_j)
                        # func connectivity for network pair
                        sub_func_network = sub_func_mat[
                            np.ix_(index_i, index_j)
                        ]
                        sub_func_network = sub_func_network[
                            ~np.isnan(sub_func_network)
                        ].flatten()
                        # print(sub_func_network)
                        # struc connectivity for network pair
                        sub_struc_network = sub_struc_mat[
                            np.ix_(index_i, index_j)
                        ]
                        sub_struc_network = sub_struc_network[
                            ~np.isnan(sub_struc_network)
                        ].flatten()
                        # print(sub_struc_network)
                        # correlation between func and struc connectivity
                        corr = np.corrcoef(sub_struc_network, sub_func_network)
                        print(corr, f"{task} {sub} {cov} {measure}")
                        network_pair_corr[network_i][network_j] = corr[0][1]

                network_pair_corr = sym_matrix_to_vec(network_pair_corr)
                result = {
                    "corr": network_pair_corr,
                    "task": task,
                    "subject": sub,
                    "cov measure": cov + " " + measure,
                }
                results.append(result)

results = pd.DataFrame(results)

for _, row in results.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"{task}_{sub}_{cov}")
    corr = vec_to_sym_matrix(corr)
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# take a mean of the network wise correlations across subjects
# gives a network wise correlation matrix for each task and cov measure
fc_sc_corr_tasks = (
    results.groupby(["task", "cov measure"])["corr"].mean().reset_index()
)
for _, row in fc_sc_corr_tasks.iterrows():
    task = row["task"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {cov}"
    output = os.path.join(output_dir, f"mean_{task}_{cov}")
    corr = vec_to_sym_matrix(corr)
    # get diagonal of corr
    diagonal = np.diag(corr)
    # barplot the diagonal values with labels on x-axis
    sns.barplot(x=le.inverse_transform(unique_labels), y=diagonal)
    plt.title(title)
    plt.xlabel("Network")
    plt.ylabel("Correlation")
    plt.savefig(output + "_boxplot.png")
    plt.close()

    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# get the mean of the network wise correlations across tasks
# gives a network wise correlation matrix for each cov measure and subject
fc_sc_corr_subjects = (
    results.groupby(["task", "cov measure"]).mean().reset_index()
)
for _, row in fc_sc_corr_subjects.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"mean_{sub}_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# get the mean of the network wise correlations across tasks and subjects
# gives a network wise correlation matrix for each cov measure
fc_sc_corr = results.groupby(["cov measure"]).mean().reset_index()
for _, row in fc_sc_corr.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"mean_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )
