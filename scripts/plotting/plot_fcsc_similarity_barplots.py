import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.plot import insert_stats_horizontal, wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

# root = "/data/parietal/store3/work/haggarwa/connectivity"
root = "/Users/himanshu/Desktop/ibc/connectivity"
results_root = os.path.join(root, "results", "wo_extra_GBU_runs")
plots_root = os.path.join(root, "plots", "wo_extra_GBU_runs")


n_parcels = 400
trim_length = None
do_hatch = False

dir_name = f"fcsc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "results.pkl")
similarity_data = pd.read_pickle(results_pkl)
similarity_data = similarity_data[similarity_data["task2"] == "SC"]

out_dir_name = (
    f"fcsc_similarity_barplots_nparcels-{n_parcels}_trim-{trim_length}"
)
output_dir = os.path.join(plots_root, out_dir_name)
os.makedirs(output_dir, exist_ok=True)

### fc-sc similarity, barplots ###
hatches = [None, "X", "\\", "/", "|"] * 8
do_hatch = False
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

for centering in similarity_data["centering"].unique():
    df = similarity_data[similarity_data["centering"] == centering]
    for how_many in ["all", "three"]:
        if how_many == "all":
            fc_measure_order = [
                "Unregularized correlation",
                "Unregularized partial correlation",
                "Ledoit-Wolf correlation",
                "Ledoit-Wolf partial correlation",
                "Graphical-Lasso correlation",
                "Graphical-Lasso partial correlation",
            ]
        elif how_many == "three":
            fc_measure_order = [
                "Unregularized correlation",
                "Ledoit-Wolf correlation",
                "Graphical-Lasso partial correlation",
            ]

        d = {"FC measure": [], "Similarity": [], "Comparison": []}
        for _, row in df.iterrows():
            corr = row["matrix"].tolist()
            n_subs = len(row["kept_subjects"])
            # reshape to square matrix
            matrix = row["matrix"].reshape((n_subs, n_subs))
            # set upper triangle to nan to keep only lower triangle
            only_lower = matrix.copy()
            only_lower[np.triu_indices_from(only_lower, k=1)] = np.nan
            # keep only non-nan values and convert to list
            corr = only_lower[~np.isnan(only_lower)].tolist()
            d["Similarity"].extend(corr)
            d["FC measure"].extend([row["measure"]] * len(corr))
            d["Comparison"].extend([row["comparison"]] * len(corr))
        d = pd.DataFrame(d)
        fig, ax = plt.subplots()

        hue_order = [
            "RestingState vs. SC",
            "LePetitPrince vs. SC",
            "Raiders vs. SC",
            "GoodBadUgly vs. SC",
            "MonkeyKingdom vs. SC",
            "Mario vs. SC",
        ]
        name = "fc_sc"
        rest_colors = sns.color_palette("tab20c")[0]
        movie_colors = sns.color_palette("tab20c")[4:7]
        mario_colors = sns.color_palette("tab20c")[8]
        lpp_colors = sns.color_palette("tab20c")[12]
        color_palette = (
            [rest_colors] + [lpp_colors] + movie_colors + [mario_colors]
        )
        ax = sns.barplot(
            x="Similarity",
            y="FC measure",
            order=fc_measure_order,
            hue="Comparison",
            orient="h",
            hue_order=hue_order,
            palette=color_palette,
            data=d,
            ax=ax,
            # errorbar=None,
        )
        wrap_labels(ax, 20)
        for i, container in enumerate(ax.containers):
            plt.bar_label(
                container,
                fmt="%.2f",
                label_type="edge",
                fontsize="x-small",
                padding=-45,
                weight="bold",
                color="white",
            )
            if do_hatch:
                # Loop over the bars
                for thisbar in container.patches:
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])

        legend = ax.legend(
            framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        if do_hatch:
            for i, handle in enumerate(legend.legend_handles):
                handle._hatch = hatches[i]

        plot_file = os.path.join(
            output_dir,
            f"similarity_{name}_{centering}_{how_many}.svg",
        )
        plot_file2 = os.path.join(
            output_dir,
            f"similarity_{name}_{centering}_{how_many}.png",
        )
        if how_many == "three":
            fig.set_size_inches(5, 5)
        else:
            fig.set_size_inches(5, 10)
        plt.savefig(plot_file, bbox_inches="tight", transparent=True)
        plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
        plt.close()
