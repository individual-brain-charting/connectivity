import pandas as pd
import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    cross_validate,
    permutation_test_score,
)
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from joblib import Parallel, delayed, dump
import time
import os
import sys


def get_nan_indices(df):
    X = np.array(df["Graphical-Lasso partial correlation"].values.tolist())
    return np.where(np.isnan(X).all(axis=1))


def hcp_subject_fingerprinting_pairwisetasks(
    df, task_1, task_2, connectivity_measure
):
    all_scores = {}
    df_task1_task2 = df[df["tasks"].isin([task_1, task_2])]
    nan_indices = get_nan_indices(df_task1_task2)
    X = np.array(df_task1_task2[connectivity_measure].values.tolist())
    y = df_task1_task2["subject_ids"].to_numpy(dtype=object)
    groups = df_task1_task2["tasks"].to_numpy(dtype=object)
    X = np.delete(X, nan_indices, axis=0)
    y = np.delete(y, nan_indices, axis=0)
    groups = np.delete(groups, nan_indices, axis=0)

    n_groups = cv_splits = len(np.unique(groups))
    # set-up cross-validation scheme
    cv = GroupKFold(n_splits=n_groups, random_state=0, shuffle=True)
    classifier = LinearSVC(max_iter=100000, dual="auto")
    # permutation test
    score, permutation_scores, pvalue = permutation_test_score(
        classifier,
        X,
        y,
        groups=groups,
        cv=cv,
        n_permutations=100,
        n_jobs=1,
        scoring="f1_macro",
        verbose=11,
        random_state=0,
    )
    return {
        "scores": score,
        "permutation_scores": permutation_scores,
        "pvalue": pvalue,
        "task1": task_1,
        "task2": task_2,
        "connectivity_measure": connectivity_measure,
    }


if __name__ == "__main__":
    root = "/data/parietal/store3/work/haggarwa/connectivity/results/"
    # root = "/Users/himanshu/Desktop/ibc/connectivity/results/"
    fc_data_path = os.path.join(
        root, "connectomes_nparcels-200_tasktype-domain_trim-None.pkl"
    )

    df = pd.read_pickle(fc_data_path)

    df = df[df["dataset"] == "HCP900"]
    df.reset_index(drop=True, inplace=True)

    tasks = [
        "HcpEmotion",
        "HcpGambling",
        "HcpLanguage",
        "HcpMotor",
        "HcpRelational",
        "HcpSocial",
        "HcpWm",
    ]

    connectivity_measures = [
        "Graphical-Lasso partial correlation",
        "Unregularized correlation",
        "Ledoit-Wolf correlation",
    ]

    if len(sys.argv) < 1:
        print(
            "Please provide task1, task2, and connectivity measure as arguments."
        )
        sys.exit(1)

    task_1 = sys.argv[1]
    task_2 = sys.argv[2]
    connectivity_measure = sys.argv[3]

    # all results
    all_results = hcp_subject_fingerprinting_pairwisetasks(
        df, task_1, task_2, connectivity_measure
    )

    # Save the results
    output_dir = "hcp_subject_fingerprinting_pairwise_tasks_permutation"
    output_path = os.path.join(root, output_dir)
    os.makedirs(output_path, exist_ok=True)
    # Save the results to a file
    dump(
        all_results,
        os.path.join(
            output_path,
            f"{task_1}_{task_2}_{connectivity_measure}.pkl",
        ),
    )

    # Print the results
    print(all_results)
