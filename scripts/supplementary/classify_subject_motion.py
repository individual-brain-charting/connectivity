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
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.fetching import get_ses_modality, get_confounds, get_niftis


def homogenize(data):
    """Keep only runs with the maximum number of samples. If multiple runs have the same number of samples, cut the longer runs to the length of the shortest run."""

    lengths = [len(run) for run in data]
    unique, counts = np.unique(lengths, return_counts=True)
    # check if all elements in counts are the same
    if len(set(counts)) == 1:
        # cut all runs to the length of the shortest run
        min_length = unique[np.argmin(counts)]
        data = [run[:min_length] for run in data]
        return np.array(data), np.ones(len(data)).astype(bool)
    else:
        # indices where count is max
        max_count_length = unique[np.argmax(counts)]
        mask = lengths == max_count_length
        return np.array(list(compress(data, mask))), mask


#### INPUTS
# plots path
plots_path = (
    "/storage/store3/work/haggarwa/connectivity/plots/classify_subjects_motion"
)
os.makedirs(plots_path, exist_ok=True)
# results directory
results_path = "/storage/store3/work/haggarwa/connectivity/results"
# output data paths
output_path = os.path.join(
    results_path,
    f"classify_motion.pkl",
)
# motion parameters
motion_path = os.path.join(
    results_path,
    f"motion_parameters.pkl",
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
    f"classify_subjects_motion.pkl",
)
for task in tqdm(tasks):
    motion_data_task = motion_data[motion_data["tasks"] == task]

    # get X, y and groups
    X = motion_data_task["motion"].tolist()
    y = np.array(motion_data_task["subject_ids"].tolist())
    groups = np.array(motion_data_task["run_labels"].tolist())

    # homogenize X, y and groups
    X, mask = homogenize(X)
    y = y[mask]
    groups = groups[mask]

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
        scoring=["accuracy", "balanced_accuracy", "f1_weighted"],
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
        scoring=["accuracy", "balanced_accuracy", "f1_weighted"],
        n_jobs=n_jobs,
        return_estimator=True,
        return_indices=True,
    )
    result = {
        "task": [task] * n_groups,
        "accuracy": cv_result["test_accuracy"].tolist(),
        "balanced_accuracy": cv_result["test_balanced_accuracy"].tolist(),
        "f1_score": cv_result["test_f1_weighted"].tolist(),
        "dummy_accuracy": cv_result_dummy["test_accuracy"].tolist(),
        "dummy_balanced_accuracy": cv_result_dummy[
            "test_balanced_accuracy"
        ].tolist(),
        "dummy_f1_score": cv_result_dummy["test_f1_weighted"].tolist(),
        "train_indices": list(cv_result["indices"]["train"]),
        "test_indices": list(cv_result["indices"]["test"]),
    }
    result = pd.DataFrame(result)
    results.append(result)

results = pd.concat(results)
results.reset_index(drop=True, inplace=True)
results.to_pickle(output_path)

results[results.select_dtypes(include=["number"]).columns] *= 100
# plot
sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")
rest_colors = sns.color_palette("tab20c")[0]
movie_colors = sns.color_palette("tab20c")[4:7]
mario_colors = sns.color_palette("tab20c")[8]
lpp_colors = sns.color_palette("tab20c")[12]
color_palette = [rest_colors] + [lpp_colors] + movie_colors + [mario_colors]
for score in ["balanced_accuracy", "f1_score"]:
    ax_score = sns.barplot(
        y="task",
        x=score,
        data=results,
        orient="h",
        palette=color_palette,
        order=tasks,
        hue="task",
    )
    # add accuracy labels on bars
    bar_label_color = "white"
    bar_label_weight = "bold"
    for i, container in enumerate(ax_score.containers):
        plt.bar_label(
            container,
            fmt="%.1f",
            label_type="edge",
            fontsize="x-small",
            padding=-45,
            weight=bar_label_weight,
            color=bar_label_color,
        )
    ax_dummy = sns.barplot(
        y="task",
        x=f"dummy_{score}",
        data=results,
        orient="h",
        order=tasks,
        facecolor=(0.8, 0.8, 0.8, 1),
        hue="task",
    )
    if score == "balanced_accuracy":
        plt.xlabel("Accuracy")
    else:
        plt.xlabel("F1 score")
    plt.ylabel("Task")
    # create legend for ax_dummy
    legend_elements = [
        Patch(
            facecolor=(0.8, 0.8, 0.8, 1),
            edgecolor="white",
            label="Chance-level",
        )
    ]
    plt.legend(
        handles=legend_elements,
        framealpha=0,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plot_file = "classify_subjects_motion"
    plt.savefig(
        os.path.join(plots_path, f"{plot_file}_{score}.png"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(plots_path, f"{plot_file}_{score}.svg"),
        bbox_inches="tight",
    )

    plt.close()
