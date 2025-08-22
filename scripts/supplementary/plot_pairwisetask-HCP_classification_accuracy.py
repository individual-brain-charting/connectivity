import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.plot import wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

# root = "/data/parietal/store3/work/haggarwa/connectivity"
root = "/Users/himanshu/Desktop/ibc/connectivity"
results_root = os.path.join(root, "results")
dir_name = "hcp_subject_fingerprinting_pairwise_tasks"
dummy_dir_name = "hcp_subject_fingerprinting_pairwise_tasks_dummy"
results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")
dummy_results_pkl = os.path.join(
    results_root, dummy_dir_name, "all_results.pkl"
)

plots_root = os.path.join(root, "plots")
output_dir = os.path.join(plots_root, dir_name)
os.makedirs(output_dir, exist_ok=True)

df = pd.read_pickle(results_pkl)
df_dummy = pd.read_pickle(dummy_results_pkl)

measures = [
    "Unregularized correlation",
    "Graphical-Lasso partial correlation",
]

df[df.select_dtypes(include=["number"]).columns] *= 100
df_dummy[df_dummy.select_dtypes(include=["number"]).columns] *= 100
for score in ["balanced_accuracies", "f1_scores"]:
    for how_many in ["all", "three"]:
        ax_score = sns.barplot(
            y="connectivity_measure",
            x=score,
            data=df,
            orient="h",
            palette=sns.color_palette()[0:1],
            order=measures,
            facecolor=(0.4, 0.4, 0.4, 1),
        )
        for i in ax_score.containers:
            plt.bar_label(
                i,
                fmt="%.1f",
                label_type="edge",
                fontsize="x-small",
                padding=-45,
                weight="bold",
                color="white",
            )
        wrap_labels(ax_score, 20)
        ax_chance = sns.barplot(
            y="connectivity_measure",
            x=score,
            data=df_dummy,
            orient="h",
            palette=sns.color_palette("pastel")[7:],
            order=measures,
            facecolor=(0.8, 0.8, 0.8, 1),
            err_kws={"ls": ":"},
        )

        if score == "balanced_accuracy":
            plt.xlabel("Accuracy")
        else:
            plt.xlabel("F1 score")
        plt.ylabel("FC measure")
        fig = plt.gcf()
        fig.set_size_inches(6, 2.5)
        plt.savefig(
            os.path.join(
                output_dir, f"hcp_pairwise_classification_{score}.png"
            ),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(
                output_dir, f"hcp_pairwise_classification_{score}.svg"
            ),
            bbox_inches="tight",
        )
        plt.close("all")
