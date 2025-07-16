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

### within or binary task classification accuracies ###
hatches = [None, "X", "\\", "/", "|"] * 8

results_root = "/Users/himanshu/Desktop/ibc/connectivity/results"
plots_root = "/Users/himanshu/Desktop/ibc/connectivity/plots"
n_parcels = 400
trim_length = None
tasktype = "natural"
do_hatch = False

dir_name = f"classification_within-thelittleprince_tasktype-{tasktype}_n_parcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")

output_dir = os.path.join(plots_root, dir_name)

os.makedirs(output_dir, exist_ok=True)
df = pd.read_pickle(results_pkl)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Subjects", "Runs"]
# tasks
tasks = ["LePetitPrince"]

df[df.select_dtypes(include=["number"]).columns] *= 100

for score in ["balanced_accuracy", "f1_macro"]:
    for clas in classify:
        df_ = df[df["classes"] == clas]
        df_.reset_index(inplace=True, drop=True)
        # rename LePetitPrince to TheLittlePrince
        df_["task_label"] = df_["task_label"].replace(
            "LePetitPrince", "TheLittlePrince"
        )
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
            if clas == "Runs":
                legend_cutoff = 1
                title = ""
                lpp_colors = sns.color_palette("tab20c")[12]
                color_palette = [lpp_colors]
                bar_label_color = "white"
                bar_label_weight = "bold"
                hue_order = ["TheLittlePrince"]
            else:
                legend_cutoff = 1
                title = ""
                lpp_colors = sns.color_palette("tab20c")[12]
                color_palette = [lpp_colors]
                bar_label_color = "white"
                bar_label_weight = "bold"
                hue_order = ["TheLittlePrince"]
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
                x="dummy_" + score,
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

            fig = plt.gcf()
            if how_many == "three":
                fig.set_size_inches(6, 3)
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
