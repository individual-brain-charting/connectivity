import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt
import seaborn as sns
from skada.datasets import DomainAwareDataset
from skada import CORALAdapter
from skada import make_da_pipeline
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    cross_validate,
    StratifiedGroupKFold,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def drop_nan_samples(X, y):
    nan_indices = np.where(np.isnan(X).all(axis=1))
    X = np.delete(X, nan_indices, axis=0)
    y = np.delete(y, nan_indices, axis=0)

    return X, y


results = "/storage/store3/work/haggarwa/connectivity/results/"
output = os.path.join(results, "across_dataset_generalize")
os.makedirs(output, exist_ok=True)

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

# dataset to keep
datasets = ["HCP900", "ibc"]

connectomes = connectomes[connectomes["dataset"].isin(datasets)]

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
    "HcpSocial",
    "HcpWm",
]
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

# Standardize the data
scaler = StandardScaler()
for cov in cov_estimators:
    for measure in measures:
        connectomes[f"{cov} {measure}"] = list(
            scaler.fit_transform(
                np.array(connectomes[f"{cov} {measure}"].tolist())
            )
        )

results = []
# loop over each cov estimator and measure
for cov in cov_estimators:
    for measure in measures:
        print(f"Processing {cov} {measure}")
        for direction in ["zero_to_one", "one_to_zero"]:
            datas = []
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
                        elif data_name == "HumanMonkeyGBU":
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
                datas.append(data)

            if direction == "zero_to_one":
                data_train = datas[0]
                data_predict = datas[1]
                id_string = f"{datasets[0]} -> {datasets[1]}"
            else:
                data_train = datas[1]
                data_predict = datas[0]
                id_string = f"{datasets[1]} -> {datasets[0]}"

            X_train = np.array(data_train[f"{cov} {measure}"].tolist())
            X_predict = np.array(data_predict[f"{cov} {measure}"].tolist())

            if tasktype == "natural":
                y_train = np.array(data_train["run_labels"].tolist())
                y_predict = np.array(data_predict["run_labels"].tolist())
            else:
                y_train = np.array(data_train["tasks"].tolist())
                y_predict = np.array(data_predict["tasks"].tolist())

            X_train, y_train = drop_nan_samples(X_train, y_train)
            X_predict, y_predict = drop_nan_samples(X_predict, y_predict)

            # Create the pipeline
            clf = LinearSVC(max_iter=10000, dual="auto").fit(X_train, y_train)
            dummy = DummyClassifier(strategy="most_frequent").fit(
                X_train, y_train
            )
            # predict
            pred = clf.predict(X_predict)
            dummy_pred = dummy.predict(X_predict)
            scores = {
                "accuracy": accuracy_score(y_predict, pred),
                "balanced_accuracy": balanced_accuracy_score(y_predict, pred),
                "f1": f1_score(y_predict, pred, average="macro"),
                "dummy_accuracy": accuracy_score(y_predict, dummy_pred),
                "dummy_balanced_accuracy": balanced_accuracy_score(
                    y_predict, dummy_pred
                ),
                "dummy_f1": f1_score(y_predict, dummy_pred, average="macro"),
                "dataset": data_name,
                "cov measure": f"{cov} {measure}",
                "direction": id_string,
            }
            print(f"\tScores for {id_string}:")
            for score in ["accuracy", "balanced_accuracy", "f1"]:
                print(
                    f"\t\t{score}: {scores[score]} | Dummy {score}: {scores[f'dummy_{score}']}"
                )
            results.append(scores)

results = pd.DataFrame(results)
results.to_csv(
    os.path.join(
        output,
        f"results_nparcels-{n_parcels}_datasets-{datasets[0]}-{datasets[1]}_trim-{trim_length}.csv",
    )
)
