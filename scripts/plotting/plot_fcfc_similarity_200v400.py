import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot import get_lower_tri_heatmap


sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

results_root = (
    "/storage/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
)
plots_root = (
    "/storage/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs/"
)
trim_length = None

out_dir_name = f"fcfc_similarity_200v400_trim-{trim_length}"
output_dir = os.path.join(plots_root, out_dir_name)
os.makedirs(output_dir, exist_ok=True)

#### comparison between distributions of fc-fc similarity values for 400 vs.
# 200 parcels
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = [
    "RestingState",
    "LePetitPrince",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]
# colors
colors = ["red", "blue"]

for cov in cov_estimators:
    for measure in measures:
        fig, ax = plt.subplots(figsize=(5, 5))
        means = []
        values = []
        for n_parcels, color in zip([200, 400], colors):
            dir_name = (
                f"fcsc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
            )
            results_pkl = os.path.join(results_root, dir_name, "results.pkl")
            similarity_data = pd.read_pickle(results_pkl)

            mask = (similarity_data["measure"] == f"{cov} {measure}") & (
                similarity_data["centering"] == "centered"
            )
            matrix = similarity_data[mask]["matrix"].to_numpy()[0]
            n_subs = len(similarity_data[mask]["kept_subjects"].to_numpy()[0])
            matrix = matrix.reshape((n_subs, n_subs))
            upper_tri = np.triu(matrix, k=1).flatten()
            lower_tri = np.tril(matrix, k=-1).flatten()
            cross_sub = np.concatenate((upper_tri, lower_tri))
            values.append(cross_sub)
            threshold = np.percentile(cross_sub, 1)
            cross_sub_thresh = cross_sub[(cross_sub > threshold)]
            ax.hist(
                cross_sub_thresh.flatten(),
                bins=50,
                label=f"{n_parcels} regions",
                color=color,
                alpha=0.5,
                density=True,
            )
            means.append(np.mean(cross_sub_thresh.flatten()))
        MWU_test = mannwhitneyu(values[0], values[1], alternative="greater")
        ax.annotate(
            f"MWU test\n200 > 400 regions:\np = {MWU_test[1]:.2e}",
            xy=(0.57, 0.83),
            xycoords="axes fraction",
            bbox={"fc": "0.8"},
            fontsize=12,
        )
        ax.axvline(
            means[0],
            color=colors[0],
            linestyle="--",
        )
        ax.axvline(
            means[1],
            color="k",
            linestyle="--",
            label="mean",
        )
        ax.axvline(
            means[1],
            color=colors[1],
            linestyle="--",
        )
        plt.legend(framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{cov} {measure}")
        plt.xlabel("FC-FC Similarity")
        plt.savefig(
            os.path.join(output_dir, f"{cov}_{measure}_hist.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{cov}_{measure}_hist.svg"),
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()
        print(
            f"Testing whether 200 > 400 for {cov} {measure}\n",
            MWU_test,
        )
