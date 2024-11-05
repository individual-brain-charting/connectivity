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

# motion results directory
motion_results_path = "/storage/store3/work/haggarwa/connectivity/results/classify_runs_motion.pkl"
motion_results = pd.read_pickle(motion_results_path)
motion_results[motion_results.select_dtypes(include=["number"]).columns] *= 100

# fc results directory
fc_results_root = (
    "/storage/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
)
n_parcels = 400
trim_length = None
tasktype = "natural"
dir_name = f"classification-within_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}"
fc_results_path = os.path.join(fc_results_root, dir_name, "all_results.pkl")
fc_results = pd.read_pickle(fc_results_path)
fc_results[fc_results.select_dtypes(include=["number"]).columns] *= 100
fc_results = fc_results[fc_results["classes"] == "Runs"]
fc_results = fc_results[
    fc_results["connectivity"] == "Unregularized correlation"
]
fc_results = fc_results.rename(
    columns={
        "task_label": "task",
        "LinearSVC_accuracy": "accuracy",
        "Dummy_accuracy": "dummy_accuracy",
        "train_sets": "train_indices",
        "test_sets": "test_indices",
    }
)
fc_results = fc_results[list(motion_results.columns)]
fc_results["feature"] = "Unregularized correlation FC"
motion_results["feature"] = "Frame-wise displacement"
results = pd.concat([fc_results, motion_results], ignore_index=True)

# plots path
plots_path = (
    "/storage/store3/work/haggarwa/connectivity/plots/classify_runs_motion"
)
os.makedirs(plots_path, exist_ok=True)

# tasks to classify
tasks = [
    "LePetitPrince",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
]

# plot
sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

for score in ["balanced_accuracy", "f1_macro"]:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax_score = sns.barplot(
        y="task",
        x=score,
        data=results,
        orient="h",
        palette="Set1",
        order=tasks,
        hue="feature",
        hue_order=["Frame-wise displacement", "Unregularized correlation FC"],
    )
    # add accuracy labels on bars
    for i, container in enumerate(ax_score.containers):
        bar_label_weight = "bold"
        if i == 1:
            bar_label_color = "white"
            padding = -35
        else:
            bar_label_color = "black"
            padding = 30
        plt.bar_label(
            container,
            fmt="%.1f",
            label_type="edge",
            fontsize="x-small",
            padding=padding,
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
        hue="feature",
        hue_order=["Frame-wise displacement", "Unregularized correlation FC"],
    )
    if score == "balanced_accuracy":
        plt.xlabel("Accuracy")
    else:
        plt.xlabel("F1 score")
    plt.ylabel("Task")
    legend = plt.legend(
        framealpha=0,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    legend_cutoff = 2
    # remove legend repetition for chance level
    for i, (text, handle) in enumerate(
        zip(legend.texts, legend.legend_handles)
    ):
        if i > legend_cutoff:
            text.set_visible(False)
            handle.set_visible(False)

        if i == legend_cutoff:
            text.set_text("Chance-level")
            handle.set_color((0.8, 0.8, 0.8, 1))

    plot_file = "classify_runs_motion"
    plt.savefig(
        os.path.join(plots_path, f"{plot_file}_{score}.png"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(plots_path, f"{plot_file}_{score}.svg"),
        bbox_inches="tight",
    )

    plt.close()
