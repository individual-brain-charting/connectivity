import os
import sys
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from itertools import compress
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.fetching import get_ses_modality, get_confounds, get_niftis


def homogenize(data):
    """Keep only runs with the maximum number of samples. If multiple runs have the same number of samples, cut the longer runs to the length of the shortest run."""

    lengths = [len(run) for run in data]
    min_length = min(lengths)
    # cut all runs to the length of the shortest run
    data = [run[:min_length] for run in data]
    return np.array(data), np.ones(len(data)).astype(bool)


#### INPUTS
# plots path
plots_path = "/data/parietal/store3/work/haggarwa/connectivity/plots/classify_subjects_fd"
os.makedirs(plots_path, exist_ok=True)
# results directory
results_path = "/data/parietal/store3/work/haggarwa/connectivity/results"
# motion parameters
motion_path = os.path.join(
    results_path,
    f"framewise_displacement.pkl",
)
motion_data = pd.read_pickle(motion_path)
motion_data = motion_data[motion_data["dataset"] == "ibc"]
# number of jobs to run in parallel
n_jobs = 10
# tasks to classify
tasks = [
    "RestingState",
    "LePetitPrince",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]

results = []

# classify subjects
# output data paths
output_path = os.path.join(
    results_path,
    f"classify_subjects_fd.pkl",
)
for task in tqdm(tasks):
    print(f"Task: {task}")
    motion_data_task = motion_data[motion_data["tasks"] == task]

    # get X, y and groups
    X = motion_data_task["motion"].tolist()
    y = np.array(motion_data_task["subject_ids"].tolist())
    groups = np.array(motion_data_task["run_labels"].tolist())

    print("Runs before homogenization:", len(X))
    # homogenize X, y and groups
    X, mask = homogenize(X)
    y = y[mask]
    groups = groups[mask]
    print("Runs after homogenization:", len(X))

    # number of groups
    n_groups = len(np.unique(groups))

    # cross-validation scheme
    cv_scheme = StratifiedGroupKFold(
        shuffle=True,
        n_splits=n_groups,
        random_state=0,
    )
    # classifier
    clf = LinearSVC(max_iter=10000, dual="auto")
    # dummy classifier
    dummy = DummyClassifier(strategy="most_frequent")

    # cross validate
    cv_result = cross_validate(
        clf,
        X,
        y,
        groups=groups,
        cv=cv_scheme,
        scoring=["accuracy", "balanced_accuracy", "f1_macro"],
        n_jobs=n_jobs,
        return_estimator=True,
        return_indices=True,
    )
    cv_result_dummy = cross_validate(
        dummy,
        X,
        y,
        groups=groups,
        cv=cv_scheme,
        scoring=["accuracy", "balanced_accuracy", "f1_macro"],
        n_jobs=n_jobs,
        return_estimator=True,
        return_indices=True,
    )
    result = {
        "task": [task] * n_groups,
        "accuracy": cv_result["test_accuracy"].tolist(),
        "balanced_accuracy": cv_result["test_balanced_accuracy"].tolist(),
        "f1_macro": cv_result["test_f1_macro"].tolist(),
        "dummy_accuracy": cv_result_dummy["test_accuracy"].tolist(),
        "dummy_balanced_accuracy": cv_result_dummy[
            "test_balanced_accuracy"
        ].tolist(),
        "dummy_f1_macro": cv_result_dummy["test_f1_macro"].tolist(),
        "train_indices": list(cv_result["indices"]["train"]),
        "test_indices": list(cv_result["indices"]["test"]),
    }
    result = pd.DataFrame(result)
    results.append(result)

results = pd.concat(results)
results.reset_index(drop=True, inplace=True)
results.to_pickle(output_path)
