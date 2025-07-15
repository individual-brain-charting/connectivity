"""CV withing IBC and external GBU datasets to get a baseline performance for
later generalization tests between the two datasets."""

import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedGroupKFold,
)
from matplotlib import pyplot as plt
import seaborn as sns

DATA_ROOT = (
    "/data/parietal/store3/work/haggarwa/connectivity/results/before_review"
)
out_root = "/data/parietal/store3/work/haggarwa/connectivity/plots"
output_dir = os.path.join(out_root, "transfer_classifier")
os.makedirs(output_dir, exist_ok=True)

# cov estimators
cov_estimators = ["Ledoit-Wolf", "Unregularized", "Graphical-Lasso"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]


for do_hist_equ in [False, True]:
    # load connectomes for external GBU
    external_connectomes = pd.read_pickle(
        os.path.join(
            DATA_ROOT,
            "external_connectivity_20240125-104121",
            "connectomes_200_compcorr.pkl",
        )
    )

    # load connectomes for IBC GBU
    IBC_connectomes = pd.read_pickle(
        os.path.join(
            DATA_ROOT,
            "connectomes_200_comprcorr",
        )
    )

    IBC_connectomes = IBC_connectomes[
        IBC_connectomes["tasks"] == "GoodBadUgly"
    ]
    IBC_connectomes = IBC_connectomes[
        IBC_connectomes["run_labels"].isin(["run-03", "run-04", "run-05"])
    ]
    IBC_connectomes.reset_index(inplace=True, drop=True)
    # rename run labels to match across datasets
    IBC_connectomes["run_labels"].replace("run-03", "1", inplace=True)
    IBC_connectomes["run_labels"].replace("run-04", "2", inplace=True)
    IBC_connectomes["run_labels"].replace("run-05", "3", inplace=True)

    external_connectomes["run_labels"].replace("run-01", "1", inplace=True)
    external_connectomes["run_labels"].replace("run-02", "2", inplace=True)
    external_connectomes["run_labels"].replace("run-03", "3", inplace=True)

    classify = ["Runs"]

    ### cv on external GBU ###
    print("\n\ncv on external GBU")
    results = []
    for clas in classify:
        for cov in cov_estimators:
            for measure in measures:
                # train
                classes = external_connectomes["run_labels"].to_numpy(
                    dtype=object
                )
                groups = external_connectomes["subject_ids"].to_numpy(
                    dtype=object
                )
                unique_groups = np.unique(groups)
                data = np.array(
                    external_connectomes[f"{cov} {measure}"].values.tolist()
                )
                classifier = LinearSVC(max_iter=1000000, dual="auto")
                dummy = DummyClassifier(strategy="most_frequent")

                # score
                accuracy = cross_val_score(
                    classifier,
                    data,
                    classes,
                    groups=groups,
                    n_jobs=len(unique_groups),
                    cv=StratifiedGroupKFold(
                        shuffle=True,
                        n_splits=len(unique_groups),
                        random_state=0,
                    ),
                    scoring="accuracy",
                )
                dummy_accuracy = cross_val_score(
                    dummy,
                    data,
                    classes,
                    groups=groups,
                    n_jobs=len(unique_groups),
                    cv=StratifiedGroupKFold(
                        shuffle=True,
                        n_splits=len(unique_groups),
                        random_state=0,
                    ),
                    scoring="accuracy",
                )
                # save results
                result = {
                    "accuracy": np.mean(accuracy),
                    "dummy_accuracy": np.mean(dummy_accuracy),
                    "cov measure": f"{cov} {measure}",
                }
                results.append(result)
                print(
                    f"{cov} {measure}: {np.mean(accuracy):.3f} |"
                    f" {np.mean(dummy_accuracy):.3f}"
                )
        results = pd.DataFrame(results)
        sns.barplot(results, y="cov measure", x="accuracy", orient="h")
        plt.axvline(np.mean(dummy_accuracy), color="k", linestyle="--")
        plt_file = os.path.join(output_dir, f"{clas}.png")
        title = "CV on external"

        plt.xlim(0, 1.05)
        plt.title(title)
        plt.savefig(
            plt_file,
            bbox_inches="tight",
        )
        plt.close()

    ### cv on IBC GBU ###
    print("\n\ncv on IBC GBU")
    results = []
    for clas in classify:
        for cov in cov_estimators:
            for measure in measures:
                # train
                classes = IBC_connectomes["run_labels"].to_numpy(dtype=object)
                groups = IBC_connectomes["subject_ids"].to_numpy(dtype=object)
                unique_groups = np.unique(groups)
                data = np.array(
                    IBC_connectomes[f"{cov} {measure}"].values.tolist()
                )
                classifier = LinearSVC(max_iter=1000000, dual="auto")
                dummy = DummyClassifier(strategy="most_frequent")

                # score
                accuracy = cross_val_score(
                    classifier,
                    data,
                    classes,
                    groups=groups,
                    n_jobs=len(unique_groups),
                    cv=StratifiedGroupKFold(
                        shuffle=True,
                        n_splits=len(unique_groups),
                        random_state=0,
                    ),
                    scoring="accuracy",
                )
                dummy_accuracy = cross_val_score(
                    dummy,
                    data,
                    classes,
                    groups=groups,
                    n_jobs=len(unique_groups),
                    cv=StratifiedGroupKFold(
                        shuffle=True,
                        n_splits=len(unique_groups),
                        random_state=0,
                    ),
                    scoring="accuracy",
                )
                # save results
                result = {
                    "accuracy": np.mean(accuracy),
                    "dummy_accuracy": np.mean(dummy_accuracy),
                    "cov measure": f"{cov} {measure}",
                }
                results.append(result)
                print(
                    f"{cov} {measure}: {np.mean(accuracy):.3f} |"
                    f" {np.mean(dummy_accuracy):.3f}"
                )
        results = pd.DataFrame(results)
        sns.barplot(results, y="cov measure", x="accuracy", orient="h")
        plt.axvline(np.mean(dummy_accuracy), color="k", linestyle="--")
        plt_file = os.path.join(output_dir, f"{clas}.png")
        title = "CV on IBC"
        plt.xlim(0, 1.05)
        plt.title(title)
        plt.savefig(
            plt_file,
            bbox_inches="tight",
        )
        plt.close()
