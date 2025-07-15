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
n_parcels = 200
trim_length = None

dir_name = f"fcsc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "results.pkl")

out_dir_name = f"scsc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
output_dir = os.path.join(plots_root, out_dir_name)
os.makedirs(output_dir, exist_ok=True)

similarity_data = pd.read_pickle(results_pkl)

similarity_data = similarity_data[
    (similarity_data["comparison"] == "SC vs. SC")
    & (similarity_data["centering"] == "uncentered")
    & (similarity_data["measure"] == "Unregularized correlation")
]
subs = similarity_data["kept_subjects"].to_numpy()[0]
n_subs = len(subs)
matrix = similarity_data["matrix"].to_numpy()[0]
matrix = matrix.reshape((n_subs, n_subs))
only_lower = matrix.copy()
only_lower[np.triu_indices_from(only_lower)] = np.nan
mean = np.nanmean(only_lower)
with sns.plotting_context("notebook"):
    get_lower_tri_heatmap(
        matrix,
        figsize=(6, 6),
        labels=subs,
        output=os.path.join(output_dir, f"scsc_similarity"),
        fontsize=15,
        triu=True,
        diag=True,
        grid=True,
        title=f"Across subject SC-SC similarity\n(mean = {mean:.2f})",
        fontweight="bold",
    )
