import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot import wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

### within or binary task classification accuracies ###
hatches = [None, "X", "\\", "/", "|"] * 8

results_root = (
    "/storage/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
)
plots_root = (
    "/storage/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs"
)
n_parcels = 400
trim_length = None
tasktype = "natural"
do_hatch = False

dir_name = f"classification-within_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")

output_dir = os.path.join(plots_root, dir_name)

os.makedirs(output_dir, exist_ok=True)
df = pd.read_pickle(results_pkl)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
    "LePetitPrince",
]
movies = tasks[1:4] + [tasks[-1]]

df[df.select_dtypes(include=["number"]).columns] *= 100

for score in ["balanced_accuracy", "f1_macro"]:
    for clas in classify:
        df_ = df[df["classes"] == clas]
        df_.reset_index(inplace=True, drop=True)
        if clas == "Runs":
            print(len(df_))
            print(df_["task_label"].unique())
            df_ = df_[df_["task_label"].isin(movies)]
            print(len(df_))
        for how_many in ["all", "three"]:
            if how_many == "all":
                order = [
                    "Unregularized correlation",
                    "Unregularized partial correlation",
                    "Ledoit-Wolf correlation",
                    "Ledoit-Wolf partial correlation",
                    "Graphical-Lasso correlation",
                    "Graphical-Lasso partial correlation",
                ]
            elif how_many == "three":
                order = [
                    "Unregularized correlation",
                    "Ledoit-Wolf correlation",
                    "Graphical-Lasso partial correlation",
                ]
            if clas == "Tasks":
                legend_cutoff = 15
                palette_init = 1
                title = ""
                rest_colors = ["#1f77b499"] + sns.color_palette("tab20c")[0:4]
                movie_colors = sns.color_palette("tab20c")[4:7]
                mario_colors = sns.color_palette("tab20c")[8:12]
                lpp_colors = sns.color_palette("tab20c")[12:15]
                color_palette = (
                    rest_colors + lpp_colors + movie_colors + mario_colors
                )
                bar_label_color = "white"
                bar_label_weight = "bold"
                hue_order = [
                    "RestingState vs LePetitPrince",
                    "RestingState vs Raiders",
                    "RestingState vs GoodBadUgly",
                    "RestingState vs MonkeyKingdom",
                    "RestingState vs Mario",
                    "LePetitPrince vs Raiders",
                    "LePetitPrince vs GoodBadUgly",
                    "LePetitPrince vs MonkeyKingdom",
                    "Raiders vs GoodBadUgly",
                    "Raiders vs MonkeyKingdom",
                    "GoodBadUgly vs MonkeyKingdom",
                    "LePetitPrince vs Mario",
                    "Raiders vs Mario",
                    "GoodBadUgly vs Mario",
                    "MonkeyKingdom vs Mario",
                ]
            elif clas == "Runs":
                legend_cutoff = 4
                palette_init = 1
                title = ""
                movie_colors = sns.color_palette("tab20c")[4:7]
                lpp_colors = sns.color_palette("tab20c")[12]
                color_palette = [lpp_colors] + movie_colors
                bar_label_color = "white"
                bar_label_weight = "bold"
                hue_order = [
                    "LePetitPrince",
                    "Raiders",
                    "GoodBadUgly",
                    "MonkeyKingdom",
                ]
            else:
                legend_cutoff = 6
                palette_init = 0
                title = ""
                rest_colors = sns.color_palette("tab20c")[0]
                movie_colors = sns.color_palette("tab20c")[4:7]
                mario_colors = sns.color_palette("tab20c")[8]
                lpp_colors = sns.color_palette("tab20c")[12]
                color_palette = (
                    [rest_colors]
                    + [lpp_colors]
                    + movie_colors
                    + [mario_colors]
                )
                bar_label_color = "white"
                bar_label_weight = "bold"
                hue_order = hue_order = [
                    "RestingState",
                    "LePetitPrince",
                    "Raiders",
                    "GoodBadUgly",
                    "MonkeyKingdom",
                    "Mario",
                ]
            ax_score = sns.barplot(
                y="connectivity",
                x=score,
                data=df_,
                orient="h",
                hue="task_label",
                palette=color_palette,
                order=order,
                hue_order=hue_order,
                # errwidth=1,
            )
            wrap_labels(ax_score, 20)
            for i, container in enumerate(ax_score.containers):
                plt.bar_label(
                    container,
                    fmt="%.1f",
                    label_type="edge",
                    fontsize="x-small",
                    padding=-45,
                    weight=bar_label_weight,
                    color=bar_label_color,
                )
                if do_hatch:
                    # Loop over the bars
                    for thisbar in container.patches:
                        # Set a different hatch for each bar
                        thisbar.set_hatch(hatches[i])
            ax_chance = sns.barplot(
                y="connectivity",
                x="dummy_" + score + "_mostfreq",
                data=df_,
                orient="h",
                hue="task_label",
                palette=sns.color_palette("pastel")[7:],
                order=order,
                facecolor=(0.8, 0.8, 0.8, 1),
                err_kws={"ls": ":"},
            )
            if score == "balanced_accuracy":
                plt.xlabel("Accuracy")
            else:
                plt.xlabel("F1 score")
            plt.ylabel("FC measure")
            plt.title(title)
            legend = plt.legend(
                framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
            )
            # remove legend repetition for chance level
            for i, (text, handle) in enumerate(
                zip(legend.texts, legend.legend_handles)
            ):
                if i > legend_cutoff:
                    text.set_visible(False)
                    handle.set_visible(False)
                else:
                    if do_hatch:
                        handle._hatch = hatches[i]

                if i == legend_cutoff:
                    text.set_text("Chance-level")

            legend.set_title("Task")
            fig = plt.gcf()
            if clas == "Tasks":
                fig.set_size_inches(6, 15)
            else:
                fig.set_size_inches(6, 6)

            plt.savefig(
                os.path.join(
                    output_dir, f"{clas}_classification_{how_many}_{score}.png"
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    output_dir, f"{clas}_classification_{how_many}_{score}.svg"
                ),
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()
