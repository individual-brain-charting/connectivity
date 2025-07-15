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

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
plots_root = (
    "/data/parietal/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs/"
)
n_parcels = 400
trim_length = None

dir_name = f"fcsc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "results.pkl")
similarity_data = pd.read_pickle(results_pkl)

out_dir_name = (
    f"fcfc_similarity_subspec_nparcels-{n_parcels}_trim-{trim_length}"
)
output_dir = os.path.join(plots_root, out_dir_name)
os.makedirs(output_dir, exist_ok=True)

cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
measures = ["correlation", "partial correlation"]
# tasks
tasks = [
    "RestingState",
    "LePetitPrince",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]
diffss = []
simss = []
for cov in cov_estimators:
    for measure in measures:
        similarity_values = np.zeros((len(tasks), len(tasks)))
        similarity_type = np.zeros((len(tasks), len(tasks)), dtype=object)
        diffs = []
        sims = []
        for centering in ["uncentered", "centered"]:
            for i, task1 in enumerate(tasks):
                for j, task2 in enumerate(tasks):
                    if i < j:
                        mask = (
                            (similarity_data["task1"] == task1)
                            & (similarity_data["task2"] == task2)
                            & (
                                similarity_data["measure"]
                                == f"{cov} {measure}"
                            )
                            & (similarity_data["centering"] == centering)
                        )
                        matrix = similarity_data[mask]["matrix"].to_numpy()[0]
                        n_subs = len(
                            similarity_data[mask]["kept_subjects"].to_numpy()[
                                0
                            ]
                        )
                        matrix = matrix.reshape((n_subs, n_subs))
                        same_sub = np.mean(np.diagonal(matrix, offset=0))
                        upper_tri = np.triu(matrix, k=1).flatten()
                        lower_tri = np.tril(matrix, k=-1).flatten()
                        cross_sub = np.concatenate((upper_tri, lower_tri))
                        cross_sub = np.mean(cross_sub)
                        similarity_values[i][j] = same_sub
                        similarity_values[j][i] = cross_sub
                        similarity_type[i][j] = ""
                        similarity_type[j][i] = ""
                        diffs.append(same_sub - cross_sub)
                        sims.extend(matrix.flatten())
                    elif i == j:
                        similarity_values[i][j] = np.nan
                        similarity_type[i][j] = ""
                    else:
                        continue
                    similarity_type[1][3] = "Within-subject\nsimilarity"
                    similarity_type[3][1] = "Across-subject\nsimilarity"
            similarity_type = pd.DataFrame(similarity_type)
            similarity_type = similarity_type.astype("str")
            similarity_annot = similarity_values.round(2)
            similarity_annot = similarity_annot.astype("str")
            # similarity_annot[1][3] = "Within\nsubs"
            # similarity_annot[3][1] = "Across\nsubs"
            with sns.plotting_context("notebook"):
                get_lower_tri_heatmap(
                    similarity_values,
                    figsize=(5, 5),
                    # cmap="RdBu_r",
                    annot=similarity_annot,
                    labels=tasks,
                    output=os.path.join(
                        output_dir, f"subspec_{cov}_{measure}_{centering}"
                    ),
                    title=f"{cov} {measure}",
                    fontsize=15,
                    vmax=0.85,
                    vmin=0.13,
                )
            if centering == "centered":
                print(
                    f"{cov} {measure}: {np.mean(diffs):.2f} +/- {np.std(diffs):.2f}"
                )
                print(f"Av sim {cov} {measure}: {np.nanmean(sims):.2f}")
                if (
                    cov == "Graphical-Lasso"
                    and measure == "partial correlation"
                ):
                    diffss.append(diffs)
                    simss.append(sims)
                elif cov == "Ledoit-Wolf" and measure == "correlation":
                    diffss.append(diffs)
                    simss.append(sims)
                elif cov == "Unregularized" and measure == "correlation":
                    diffss.append(diffs)
                    simss.append(sims)
                else:
                    continue
print(
    "\n\n***Testing whether difference between within sub similarity and "
    "across sub similarity is greater in Graphical-Lasso partial corr than "
    "corr measures***"
)
print(
    "Graphical-Lasso partial corr > Ledoit-Wolf corr\n",
    mannwhitneyu(
        diffss[0],
        diffss[1],
        alternative="greater",
    ),
)
print(
    "Graphical-Lasso partial corr > Unregularized corr\n",
    mannwhitneyu(
        diffss[0],
        diffss[2],
        alternative="greater",
    ),
)
print(
    "\n\n***Testing whether similarity values for Graphical-Lasso partial "
    "corr are greater than for corr measures***"
)
print(
    "Graphical-Lasso partial corr > Ledoit-Wolf corr\n",
    mannwhitneyu(
        simss[0],
        simss[1],
        alternative="greater",
    ),
)
print(
    "Graphical-Lasso partial corr > Unregularized corr\n",
    mannwhitneyu(
        simss[0],
        simss[2],
        alternative="greater",
    ),
)
