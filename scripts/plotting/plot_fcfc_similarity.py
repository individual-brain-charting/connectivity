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
n_parcels = 200
trim_length = None

dir_name = f"fcsc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "results.pkl")
similarity_data = pd.read_pickle(results_pkl)

out_dir_name = f"fcfc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
output_dir = os.path.join(plots_root, out_dir_name)
os.makedirs(output_dir, exist_ok=True)


### fc-fc similarity, sub-spec matrices
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
for cov in cov_estimators:
    for measure in measures:
        similarity_values = np.zeros((len(tasks), len(tasks)))
        similarity_tasks = np.zeros((len(tasks), len(tasks)), dtype=object)
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i == j:
                    similarity = 1
                elif i > j:
                    similarity = similarity_values[j][i]
                else:
                    mask = (
                        (similarity_data["task1"] == task1)
                        & (similarity_data["task2"] == task2)
                        & (similarity_data["measure"] == f"{cov} {measure}")
                        & (similarity_data["centering"] == "uncentered")
                    )
                    matrix = similarity_data[mask]["matrix"].to_numpy()[0]
                    similarity = np.mean(matrix)
                similarity_values[i][j] = similarity
                similarity_tasks[i][j] = (task1, task2)
        with sns.plotting_context("notebook"):
            get_lower_tri_heatmap(
                similarity_values,
                # cmap="Reds",
                figsize=(5, 5),
                labels=tasks,
                output=os.path.join(output_dir, f"similarity_{cov}_{measure}"),
                triu=True,
                diag=True,
                tril=False,
                title=f"{cov} {measure}",
                fontsize=15,
            )
