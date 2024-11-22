"""This script fits classifiers to full data and
 plots the classifier coefficients"""

import os
import pandas as pd
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed
from sklearn import preprocessing
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.svm import LinearSVC

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot import get_clas_cov_measure
from utils.fc_classification import drop_nan_samples


def fit_classifier(clas, cov, measure, func_data, output_dir):
    weights_file = os.path.join(
        output_dir, f"{clas}_{cov} {measure}_weights.npy"
    )
    class_labels_file = os.path.join(
        output_dir, f"{clas}_{cov} {measure}_labels.npy"
    )
    if os.path.exists(weights_file):
        print(f"skipping {cov} {measure}, already done")
        pass
    else:
        if clas == "Tasks":
            classes = func_data["tasks"].to_numpy(dtype=object)
        elif clas == "Subjects":
            classes = func_data["subject_ids"].to_numpy(dtype=object)
        elif clas == "Runs":
            func_data["run_task"] = (
                func_data["run_labels"] + "_" + func_data["tasks"]
            )
            classes = func_data["run_task"].to_numpy(dtype=object)
        data = np.array(func_data[f"{cov} {measure}"].values.tolist())
        data, classes, _ = drop_nan_samples(
            data, classes, None, None, f"{cov} {measure}", clas
        )
        classifier = LinearSVC(max_iter=100000, dual="auto").fit(data, classes)
        np.save(weights_file, classifier.coef_)
        np.save(class_labels_file, classifier.classes_)


if __name__ == "__main__":
    results_root = (
        "/storage/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
    )
    weights_root = (
        "/storage/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
    )
    n_parcels = 400
    trim_length = None

    connectome_pkl = f"connectomes_nparcels-{n_parcels}_tasktype-natural_trim-{trim_length}.pkl"
    connectome_pkl = os.path.join(results_root, connectome_pkl)

    out_dir_name = f"weights_taskwise_nparcels-{n_parcels}_trim-{trim_length}"
    output_dir = os.path.join(weights_root, out_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_pickle(connectome_pkl)
    df = df[df["dataset"] == "ibc"]

    cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
    measures = ["correlation", "partial correlation"]
    # classify = ["Tasks", "Subjects", "Runs"]
    classify = ["Tasks"]
    x = Parallel(n_jobs=20, verbose=11)(
        delayed(fit_classifier)(clas, cov, measure, df, output_dir=output_dir)
        for clas, cov, measure in get_clas_cov_measure(
            classify, cov_estimators, measures
        )
    )
