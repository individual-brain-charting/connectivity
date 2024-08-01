"""This script creates 2D UMAP representations of connectomes with different run 
lengths, to assess the effect of run length on the classification of 
connectomes"""

import umap
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt
import seaborn as sns


results = "/storage/store3/work/haggarwa/connectivity/results/"
plots = "/storage/store3/work/haggarwa/connectivity/plots/umap_runlength"
os.makedirs(plots, exist_ok=True)

# variables first set of connectomes
n_parcels1 = 200
tasktype1 = "natural"
trim_length1 = 293

# variables second set of connectomes
n_parcels2 = 200
tasktype2 = "natural"
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
datasets = ["ibc"]
connectomes1 = connectomes1[connectomes1["dataset"].isin(datasets)]
connectomes2 = connectomes2[connectomes2["dataset"].isin(datasets)]

# tasks to keep
tasks = [
    "RestingState",
    # "Raiders",
    # "GoodBadUgly",
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
connectomes1 = connectomes1[connectomes1["tasks"].isin(tasks)]
connectomes2 = connectomes2[connectomes2["tasks"].isin(tasks)]

# combine both sets of connectomes
connectomes = pd.concat([connectomes1, connectomes2], ignore_index=True)
connectomes.reset_index(inplace=True, drop=True)

# cov estimators
cov_estimators = [
    "Graphical-Lasso",
    "Unregularized",
    "Ledoit-Wolf",
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
            # rename None to full
            connectomes_dropped["trim_length"].replace(
                {None: "full"}, inplace=True
            )
            # change type to str
            connectomes_dropped["trim_length"] = connectomes_dropped[
                "trim_length"
            ].astype(str)
            trim_length = np.array(connectomes_dropped["trim_length"].tolist())

            # get task names
            task = np.array(connectomes_dropped["tasks"].tolist())

            # subject names
            subject = np.array(connectomes_dropped["subject_ids"].tolist())
        else:
            # rename None to full
            connectomes["trim_length"].replace({None: "full"}, inplace=True)
            # change type to str
            connectomes["trim_length"] = connectomes["trim_length"].astype(str)
            trim_length = np.array(connectomes["trim_length"].tolist())
            # get task names
            task = np.array(connectomes["tasks"].tolist())
            # subject names
            subject = np.array(connectomes["subject_ids"].tolist())

        print(connectome_data.shape)
        print(trim_length.shape)

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
            hue=trim_length,
            style=subject,
            palette="viridis",
            legend="full",
        )
        plt.title(
            f"UMAP of {cov} {measure} connectomes with different run lengths"
        )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plots,
                f"umap_comparerunlength_{cov}_{measure}_{tasktype1}_{tasktype2}_{trim_length1}_{trim_length2}.png",
            )
        )
        plt.close()
