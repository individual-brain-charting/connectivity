import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot import insert_stats, wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.6)

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
plots_root = "/data/parietal/store3/work/haggarwa/connectivity/plots/graphical_abstract/"
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
cov_estimators = ["Graphical-Lasso", "Unregularized"]
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
    df.reset_index(inplace=True, drop=True)
    df["measure"] = df["measure"].map(
        {
            "Unregularized correlation": "Correlation",
            "Graphical-Lasso partial correlation": "Partial correlation",
        }
    )

    fc_measure_order = [
        "Correlation",
        "Partial correlation",
    ]

    d = {"FC measure": [], "Similarity": [], "Comparison": []}
    for _, row in df.iterrows():
        corr = row["matrix"].tolist()
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
    wrap_labels(ax, 10)
    for i, container in enumerate(ax.containers):
        plt.bar_label(
            container,
            fmt="%.2f",
            label_type="edge",
            fontsize="x-small",
            padding=-75,
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
        f"similarity_{name}_{centering}.svg",
    )
    plot_file2 = os.path.join(
        output_dir,
        f"similarity_{name}_{centering}.png",
    )

    fig.set_size_inches(5, 5)
    plt.ylabel("")
    plt.title("FC-SC similarity")
    plt.savefig(plot_file, bbox_inches="tight", transparent=True)
    plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
    plt.close()
