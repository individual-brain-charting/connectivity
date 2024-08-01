"""This script creates 2D UMAP representations of connectomes from same task but different dataset, to assess the effect of dataset on the classification of connectomes"""

import umap
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt
import seaborn as sns


results = "/storage/store3/work/haggarwa/connectivity/results/"
plots = "/storage/store3/work/haggarwa/connectivity/plots/umap_dataset"
os.makedirs(plots, exist_ok=True)

# variables first set of connectomes
n_parcels1 = 200
tasktype1 = "natural"
trim_length1 = None

# variables second set of connectomes
n_parcels2 = 200
tasktype2 = "domain"
trim_length2 = None

# load both sets of connectomes
connectomes1 = pd.read_pickle(
    os.path.join(
        results,
        f"connectomes_nparcels-{n_parcels1}_tasktype-{tasktype1}_trim-{trim_length1}.pkl",
    )
)
connectomes2 = pd.read_pickle(
    os.path.join(
        results,
        f"connectomes_nparcels-{n_parcels2}_tasktype-{tasktype2}_trim-{trim_length2}.pkl",
    )
)
# label with trim length
connectomes1["trim_length"] = trim_length1
connectomes2["trim_length"] = trim_length2

# dataset to keep
datasets = ["ibc", "HCP900"]
connectomes1 = connectomes1[connectomes1["dataset"].isin(datasets)]
connectomes2 = connectomes2[connectomes2["dataset"].isin(datasets)]

# tasks to keep
tasks = [
    # "RestingState",
    # "Raiders",
    # "GoodBadUgly",
    # "MonkeyKingdom",
    # "Mario",
    # "LePetitPrince",
    # "ArchiStandard",
    # "ArchiSpatial",
    # "ArchiSocial",
    # "ArchiEmotional",
    "HcpEmotion",
    "HcpGambling",
    "HcpLanguage",
    "HcpMotor",
    "HcpRelational",
    # "lppFR",
    "EMOTION",
    "GAMBLING",
    "LANGUAGE",
    "MOTOR",
    "RELATIONAL",
]
connectomes1 = connectomes1[connectomes1["tasks"].isin(tasks)]
connectomes2 = connectomes2[connectomes2["tasks"].isin(tasks)]


# add tasktype column
connectomes1["tasktype"] = [tasktype1] * len(connectomes1)
connectomes2["tasktype"] = [tasktype2] * len(connectomes2)

# combine both sets of connectomes
connectomes = pd.concat([connectomes1, connectomes2], ignore_index=True)
connectomes.reset_index(inplace=True, drop=True)

# cov estimators
cov_estimators = [
    "Unregularized",
    "Ledoit-Wolf",
    "Graphical-Lasso",
]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]

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
            # get tasktype names
            tasktype = np.array(connectomes_dropped["tasktype"].tolist())
            # get task names
            task = np.array(connectomes_dropped["tasks"].tolist())
            # subject names
            subject = np.array(connectomes_dropped["subject_ids"].tolist())
            # dataset names
            dataset = np.array(connectomes_dropped["dataset"].tolist())
        else:
            # get tasktype names
            tasktype = np.array(connectomes["tasktype"].tolist())
            # get task names
            task = np.array(connectomes["tasks"].tolist())
            # subject names
            subject = np.array(connectomes["subject_ids"].tolist())
            # dataset names
            dataset = np.array(connectomes["dataset"].tolist())

        print(connectome_data.shape)

        # standardize the connectomes
        scaler = StandardScaler()
        connectome_data = scaler.fit_transform(connectome_data)

        # # reduce the dimensionality of the connectomes using UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(connectome_data)

        # # plot the UMAP representation
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=dataset,
            style=task,
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
                f"umap_comparetasktype_{cov}_{measure}_{datasets[0]}_{datasets[1]}_{trim_length1}_{trim_length2}.png",
            )
        )
        plt.close()
