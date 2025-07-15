"""This script creates 2D UMAP representations of connectomes from same task but different dataset, to assess the effect of dataset on the classification of connectomes"""

import umap
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt
import seaborn as sns

results = "/data/parietal/store3/work/haggarwa/connectivity/results/"

# variables first set of connectomes
n_parcels = 200
tasktype = "natural"
trim_length = None

# load both sets of connectomes
connectomes = pd.read_pickle(
    os.path.join(
        results,
        f"connectomes_nparcels-{n_parcels}_tasktype-{tasktype}_trim-{trim_length}.pkl",
    )
)

# label with trim length
connectomes["trim_length"] = trim_length

# dataset to keep
datasets = ["HumanMonkeyGBU", "ibc"]
connectomes = connectomes[connectomes["dataset"].isin(datasets)]

# tasks to keep
tasks = [
    # "RestingState",
    # "Raiders",
    "GoodBadUgly",
    # "MonkeyKingdom",
    # "Mario",
    # "LePetitPrince",
    # "ArchiStandard",
    # "ArchiSpatial",
    # "ArchiSocial",
    # "ArchiEmotional",
    # "HcpEmotion",
    # "HcpGambling",
    # "HcpLanguage",
    # "HcpMotor",
    # "HcpRelational",
]
connectomes = connectomes[connectomes["tasks"].isin(tasks)]

# add tasktype column
connectomes["tasktype"] = [tasktype] * len(connectomes)

# reset index
connectomes.reset_index(inplace=True, drop=True)

# we only have three runs in common across ibc and HumanMonkeyGBU
# so we will only keep those runs
if "ibc" in datasets and "HumanMonkeyGBU" in datasets:
    if "GoodBadUgly" in tasks:
        # only keep run_labels run-03, run-04, run-05 for dataset ibc
        ibc_connectomes = connectomes.loc[
            np.where(
                (
                    connectomes["run_labels"].isin(
                        ["run-03", "run-04", "run-05"]
                    )
                )
                & (connectomes["dataset"] == "ibc")
            )
        ]

        # only keep run_labels run-01, run-02, run-03 for dataset
        # HumanMonkeyGBU
        HumanMonkeyGBU_connectomes = connectomes.loc[
            np.where(
                (
                    connectomes["run_labels"].isin(
                        ["run-01", "run-02", "run-03"]
                    )
                )
                & (connectomes["dataset"] == "HumanMonkeyGBU")
            )
        ]
        # rename run labels to match across datasets
        HumanMonkeyGBU_connectomes["run_labels"] = HumanMonkeyGBU_connectomes[
            "run_labels"
        ].map({"run-01": "run-03", "run-02": "run-04", "run-03": "run-05"})
        connectomes = pd.concat([ibc_connectomes, HumanMonkeyGBU_connectomes])
        del ibc_connectomes, HumanMonkeyGBU_connectomes

# cov estimators
cov_estimators = [
    "Unregularized",
    "Ledoit-Wolf",
    "Graphical-Lasso",
]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]

plots = f"/data/parietal/store3/work/haggarwa/connectivity/plots/umap_dataset/{datasets[0]}_{datasets[1]}"
os.makedirs(plots, exist_ok=True)

# loop over each cov estimator and measure
for cov in cov_estimators:
    for measure in measures:
        print(f"Processing {cov} {measure}")
        # there are some runs with missing Graphical Lasso connectomes
        # get their indices and drop them
        connectome_data = np.array(connectomes[f"{cov} {measure}"].tolist())
        if cov == "Graphical-Lasso":
            nan_indices = np.where(np.isnan(connectome_data).all(axis=1))
            connectomes_dropped = connectomes.drop(index=nan_indices[0])
            print(
                f"Dropped {len(nan_indices[0])} missing Graph Lasso connectomes"
            )
            connectome_data = np.array(
                connectomes_dropped[f"{cov} {measure}"].tolist()
            )

        print(connectome_data.shape)

        # standardize the connectomes
        scaler = StandardScaler()
        connectome_data = scaler.fit_transform(connectome_data)

        # # reduce the dimensionality of the connectomes using UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(connectome_data)

        if cov == "Graphical-Lasso":
            connectomes_dropped["umap1"] = embedding[:, 0]
            connectomes_dropped["umap2"] = embedding[:, 1]
        else:
            connectomes["umap1"] = embedding[:, 0]
            connectomes["umap2"] = embedding[:, 1]

        # # plot the UMAP representation
        sns.scatterplot(
            data=(
                connectomes_dropped
                if cov == "Graphical-Lasso"
                else connectomes
            ),
            x="umap1",
            y="umap2",
            hue="dataset",
            hue_order=datasets,
            style="run_labels" if tasktype == "natural" else "tasks",
            palette="viridis",
            legend="full",
        )
        plt.title(
            f"UMAP of {cov} {measure} connectomes from different task types"
        )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plots,
                f"umap_comparetasktype_{cov}_{measure}_{datasets[0]}_{datasets[1]}_{trim_length}.png",
            )
        )
        plt.close()
