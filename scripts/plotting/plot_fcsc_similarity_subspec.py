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
sns.set_context("talk")


results_root = (
    "/storage/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
)
plots_root = (
    "/storage/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs/"
)
n_parcels = 200
trim_length = None
do_hatch = False

dir_name = f"fcsc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "results.pkl")
similarity_data = pd.read_pickle(results_pkl)
similarity_data = similarity_data[similarity_data["task2"] == "SC"]

out_dir_name = (
    f"fcsc_similarity_subspec_nparcels-{n_parcels}_trim-{trim_length}"
)
output_dir = os.path.join(plots_root, out_dir_name)
os.makedirs(output_dir, exist_ok=True)

### fc-sc similarity, sub-spec
hatches = [None, "X", "\\", "/", "|"] * 8
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
    for cov in cov_estimators:
        for measure in measures:
            df = similarity_data[
                (similarity_data["centering"] == centering)
                & (similarity_data["measure"] == f"{cov} {measure}")
            ]
            for test in ["t", "mwu"]:
                d = {
                    "Comparison": [],
                    "FC measure": [],
                    "Task vs. SC": [],
                    "Similarity": [],
                }
                p_values = {}
                for _, row in df.iterrows():
                    n_subs = len(row["kept_subjects"])
                    corr = row["matrix"].reshape(n_subs, n_subs)
                    same_sub = np.diagonal(corr, offset=0).tolist()
                    upper_tri = corr[np.triu_indices_from(corr, k=1)].tolist()
                    lower_tri = corr[np.tril_indices_from(corr, k=-1)].tolist()
                    cross_sub = upper_tri + lower_tri
                    d["Comparison"].extend(
                        [f"{row['comparison']} Within Subject"] * len(same_sub)
                        + [f"{row['comparison']} Across Subject"]
                        * len(cross_sub)
                    )
                    d["Task vs. SC"].extend(
                        [f"{row['comparison']}"]
                        * (len(same_sub) + len(cross_sub))
                    )
                    d["FC measure"].extend(
                        [row["measure"]] * (len(same_sub) + len(cross_sub))
                    )
                    d["Similarity"].extend(same_sub + cross_sub)
                    p_values[row["comparison"]] = row[f"p_value_{test}"]
                d = pd.DataFrame(d)
                hue_order = [
                    "RestingState vs. SC Across Subject",
                    "RestingState vs. SC Within Subject",
                    "LePetitPrince vs. SC Across Subject",
                    "LePetitPrince vs. SC Within Subject",
                    "Raiders vs. SC Across Subject",
                    "Raiders vs. SC Within Subject",
                    "GoodBadUgly vs. SC Across Subject",
                    "GoodBadUgly vs. SC Within Subject",
                    "MonkeyKingdom vs. SC Across Subject",
                    "MonkeyKingdom vs. SC Within Subject",
                    "Mario vs. SC Across Subject",
                    "Mario vs. SC Within Subject",
                ]
                tasks = [
                    "RestingState vs. SC",
                    "LePetitPrince vs. SC",
                    "Raiders vs. SC",
                    "GoodBadUgly vs. SC",
                    "MonkeyKingdom vs. SC",
                    "Mario vs. SC",
                ]
                rest_colors = sns.color_palette("tab20c")[0]
                lpp_colors = sns.color_palette("tab20c")[12]
                movie_1 = sns.color_palette("tab20c")[4]
                movie_2 = sns.color_palette("tab20c")[5]
                movie_3 = sns.color_palette("tab20c")[6]
                mario_colors = sns.color_palette("tab20c")[8]
                color_palette = list(
                    [rest_colors] * 2
                    + [lpp_colors] * 2
                    + [movie_1] * 2
                    + [movie_2] * 2
                    + [movie_3] * 2
                    + [mario_colors] * 2
                )
                fig = plt.figure()
                ax1 = plt.subplot2grid((1, 15), (0, 0), colspan=12)
                ax2 = plt.subplot2grid((1, 15), (0, -3))
                sns.violinplot(
                    x="Similarity",
                    y="Comparison",
                    order=hue_order,
                    hue="Comparison",
                    hue_order=hue_order,
                    palette=color_palette,
                    orient="h",
                    data=d,
                    ax=ax1,
                    split=True,
                    inner=None,
                )
                violin_patches = []
                for i, patch in enumerate(ax1.get_children()):
                    if isinstance(patch, mpl.collections.PolyCollection):
                        violin_patches.append(patch)
                for i, patch in enumerate(violin_patches):
                    if i % 2 == 0:
                        # Loop over the bars
                        patch.set_hatch(hatches[1])
                        patch.set_edgecolor("k")

                legend_elements = [
                    mpl.patches.Patch(
                        facecolor="white",
                        edgecolor="k",
                        label="Across Subject",
                    ),
                    mpl.patches.Patch(
                        facecolor="white",
                        edgecolor="k",
                        label="Within Subject",
                    ),
                ]
                legend_elements[0].set_hatch(hatches[1])
                ax1.legend(
                    framealpha=0,
                    loc="center left",
                    bbox_to_anchor=(1.2, 0.5),
                    handles=legend_elements,
                )

                for i, task in enumerate(tasks):
                    index = abs((i - len(p_values)) - 1)
                    insert_stats(
                        ax2,
                        p_values[task],
                        d["Similarity"],
                        loc=[index + 0.2, index + 0.6],
                        x_n=len(p_values),
                    )
                ax1.set_yticks(np.arange(0, 12, 2) + 0.5, tasks)
                ax1.set_ylabel("Task vs. SC")
                ax1.set_xlabel("Similarity")
                plt.title(f"{cov} {measure}", loc="right", x=-1, y=1.05)
                plot_file = os.path.join(
                    output_dir,
                    f"{cov}_{measure}_{centering}_{test}_box.svg",
                )
                plot_file2 = os.path.join(
                    output_dir,
                    f"{cov}_{measure}_{centering}_{test}_box.png",
                )
                plt.savefig(plot_file, bbox_inches="tight", transparent=True)
                plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
                plt.close()
