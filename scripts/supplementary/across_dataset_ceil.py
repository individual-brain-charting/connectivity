import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    cross_validate,
    StratifiedGroupKFold,
)

results = "/data/parietal/store3/work/haggarwa/connectivity/results/"

# variables first set of connectomes
n_parcels = 200
tasktype = "domain"
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

# dataset to task mapping
dataset_task = {
    "HCP900": [
        "HcpEmotion",
        "HcpGambling",
        "HcpLanguage",
        "HcpMotor",
        "HcpRelational",
        "HcpSocial",
        "HcpWm",
    ],
    "archi": [
        "ArchiStandard",
        "ArchiSpatial",
        "ArchiSocial",
        "ArchiEmotional",
    ],
    "thelittleprince": ["LePetitPrince"],
    "HumanMonkeyGBU": ["GoodBadUgly"],
}

# dataset to keep
datasets = ["HCP900", "ibc"]
connectomes = connectomes[connectomes["dataset"].isin(datasets)]

# tasks to keep
tasks = dataset_task[datasets[0]]
connectomes = connectomes[connectomes["tasks"].isin(tasks)]

# add tasktype column
connectomes["tasktype"] = [tasktype] * len(connectomes)

# reset index
connectomes.reset_index(inplace=True, drop=True)

# cov estimators
cov_estimators = [
    "Unregularized",
    "Ledoit-Wolf",
    "Graphical-Lasso",
]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]

results = []
# loop over each cov estimator and measure
for cov in cov_estimators:
    for measure in measures:
        print(f"Processing {cov} {measure}")
        for data_name in datasets:
            # we only have three runs in common across ibc and HumanMonkeyGBU
            # so we will only keep those runs
            if "ibc" in datasets and "HumanMonkeyGBU" in datasets:
                if "GoodBadUgly" in tasks:
                    if data_name == "ibc":
                        # only keep run_labels run-03, run-04, run-05 for dataset ibc
                        data = connectomes.loc[
                            np.where(
                                (
                                    connectomes["run_labels"].isin(
                                        ["run-03", "run-04", "run-05"]
                                    )
                                )
                                & (connectomes["dataset"] == data_name)
                            )
                        ]
                    if data_name == "HumanMonkeyGBU":
                        # only keep run_labels run-01, run-02, run-03 for dataset
                        # HumanMonkeyGBU
                        data = connectomes.loc[
                            np.where(
                                (
                                    connectomes["run_labels"].isin(
                                        ["run-01", "run-02", "run-03"]
                                    )
                                )
                                & (connectomes["dataset"] == data_name)
                            )
                        ]
                        # rename run labels to match across datasets
                        data["run_labels"] = data["run_labels"].map(
                            {
                                "run-01": "run-03",
                                "run-02": "run-04",
                                "run-03": "run-05",
                            }
                        )
            else:
                data = connectomes.loc[
                    np.where(connectomes["dataset"] == data_name)
                ]

            X = np.array(data[f"{cov} {measure}"].tolist())
            if tasktype == "natural":
                y = np.array(data["run_labels"].tolist())
            else:
                y = np.array(data["tasks"].tolist())
            groups = np.array(data["subject_ids"].tolist())
            n_groups = np.unique(groups).shape[0]

            clf = LinearSVC(max_iter=10000, dual="auto")
            dummy = DummyClassifier(strategy="most_frequent")
            cv = StratifiedGroupKFold(
                shuffle=True,
                n_splits=n_groups,
                random_state=0,
            )

            # cross validate
            result = cross_validate(
                clf,
                X,
                y,
                cv=cv,
                n_jobs=n_groups,
                params={"groups": groups},
            )
            dummy_result = cross_validate(
                dummy,
                X,
                y,
                cv=cv,
                n_jobs=n_groups,
                params={"groups": groups},
            )
            print(
                f"Score on {data_name}: {np.mean(result["test_score"])} | Dummy: {np.mean(dummy_result["test_score"])}"
            )
            results.append(
                {
                    "dataset": data_name,
                    "cov measure": f"{cov} {measure}",
                    "score": np.mean(result["test_score"]),
                    "dummy_score": np.mean(dummy_result["test_score"]),
                }
            )
