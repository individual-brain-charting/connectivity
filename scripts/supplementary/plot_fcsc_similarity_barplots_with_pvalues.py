import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from scipy import stats
from scipy.stats import false_discovery_control


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

        # calculate p-values for comparison between tasks within each fc measure
        p_values_table = {
            "centering": [],
            "task1": [],
            "task2": [],
            "ttest": [],
            "mwu": [],
            "fc_measure": [],
        }
        for fc_measure in d["FC measure"].unique():
            task_comparisons = d[d["FC measure"] == fc_measure][
                "Comparison"
            ].unique()
            for i, task1 in enumerate(task_comparisons):
                for task2 in task_comparisons[i + 1 :]:
                    subset = d[d["FC measure"] == fc_measure]
                    subset = subset[
                        (subset["Comparison"] == task1)
                        | (subset["Comparison"] == task2)
                    ]
                    p_value_ttest = false_discovery_control(
                        stats.ttest_ind(
                            subset[subset["Comparison"] == task1][
                                "Similarity"
                            ],
                            subset[subset["Comparison"] == task2][
                                "Similarity"
                            ],
                            alternative="two-sided",
                        ).pvalue
                    )
                    p_value_mwu = false_discovery_control(
                        stats.mannwhitneyu(
                            subset[subset["Comparison"] == task1][
                                "Similarity"
                            ],
                            subset[subset["Comparison"] == task2][
                                "Similarity"
                            ],
                            alternative="two-sided",
                        ).pvalue
                    )
                    p_values_table["centering"].append(centering)
                    p_values_table["task1"].append(task1)
                    p_values_table["task2"].append(task2)
                    p_values_table["ttest"].append(p_value_ttest)
                    p_values_table["mwu"].append(p_value_mwu)
                    p_values_table["fc_measure"].append(fc_measure)
        p_values_table = pd.DataFrame(p_values_table)

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
            framealpha=0,
            loc="upper center",
            bbox_to_anchor=(2, 0.75),
            ncol=1,
        )
        if do_hatch:
            for i, handle in enumerate(legend.legend_handles):
                handle._hatch = hatches[i]

        # Group p-values by FC measure to avoid overlapping lines
        p_values_by_measure = {}
        for _, row in p_values_table.iterrows():
            if row["task1"] == "SC vs. SC" or row["task2"] == "SC vs. SC":
                continue
            if row["fc_measure"] not in fc_measure_order:
                continue

            fc_measure = row["fc_measure"]
            if fc_measure not in p_values_by_measure:
                p_values_by_measure[fc_measure] = []
            p_values_by_measure[fc_measure].append(row)

        # Add statistics for each FC measure group
        for fc_measure, rows in p_values_by_measure.items():
            fc_measure_idx = fc_measure_order.index(fc_measure)

            # Sort by task comparison to ensure consistent ordering
            rows = sorted(
                rows,
                key=lambda x: (
                    hue_order.index(x["task1"]),
                    hue_order.index(x["task2"]),
                ),
            )

            for offset_level, row in enumerate(rows):
                # Get indices of the two tasks being compared
                task1_idx = hue_order.index(row["task1"])
                task2_idx = hue_order.index(row["task2"])

                # Calculate the actual y-positions of the bars for horizontal plot
                n_hues = len(hue_order)
                bar_width = 0.8 / n_hues
                center_y = fc_measure_idx
                y1 = center_y + (task1_idx - (n_hues - 1) / 2) * bar_width
                y2 = center_y + (task2_idx - (n_hues - 1) / 2) * bar_width

                insert_stats_horizontal(
                    ax, row["mwu"], d["Similarity"], [y1, y2], offset_level
                )

        plot_file = os.path.join(
            output_dir,
            f"similarity_{name}_{centering}_{how_many}_with_pvalues.svg",
        )
        plot_file2 = os.path.join(
            output_dir,
            f"similarity_{name}_{centering}_{how_many}_with_pvalues.png",
        )
        if how_many == "three":
            fig.set_size_inches(
                8, 6
            )  # Increased height for top legend and width for stats
        else:
            fig.set_size_inches(
                12, 12
            )  # Increased height for top legend and width for stats
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Make room for legend at top
        plt.savefig(plot_file, bbox_inches="tight", transparent=True)
        plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
        plt.close()
