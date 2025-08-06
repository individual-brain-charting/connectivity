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

# root = "/data/parietal/store3/work/haggarwa/connectivity"
root = "/Users/himanshu/Desktop/ibc/connectivity"
results_root = os.path.join(root, "results")
plots_root = os.path.join(root, "plots")
output_dir = os.path.join(
    plots_root, "fc_acrosstask_classification_400_var-lowpass"
)
os.makedirs(output_dir, exist_ok=True)


dir_name = "fc_acrosstask_classification_400_lowpass-0.5_20250805-135040"
pkl_05 = os.path.join(results_root, dir_name, "all_results.pkl")

dir_name = "fc_acrosstask_classification_400_lowpass-0.1_20250805-142044"
pkl_01 = os.path.join(results_root, dir_name, "all_results.pkl")

dir_name = "wo_extra_GBU_runs/classification-across_tasktype-natural_nparcels-400_trim-None"
pkl_02 = os.path.join(results_root, dir_name, "all_results.pkl")

df_05 = pd.read_pickle(pkl_05)
df_05[df_05.select_dtypes(include=["number"]).columns] *= 100
df_05["lowpass"] = "Nyquist limit (0.249)"
# add suffix "mostfreq" to all dummy columns
df_05 = df_05.rename(
    columns=lambda x: x + "_mostfreq" if x.startswith("dummy_") else x
)
df_01 = pd.read_pickle(pkl_01)
df_01[df_01.select_dtypes(include=["number"]).columns] *= 100
df_01["lowpass"] = 0.1
# add suffix "mostfreq" to all dummy columns
df_01 = df_01.rename(
    columns=lambda x: x + "_mostfreq" if x.startswith("dummy_") else x
)

df_02 = pd.read_pickle(pkl_02)
df_02[df_02.select_dtypes(include=["number"]).columns] *= 100
df_02["lowpass"] = "Original (0.2)"

# combine dataframes
df = pd.concat([df_05, df_01, df_02], axis=0)
df.reset_index(inplace=True, drop=True)

df = df[
    df["connectivity"].isin(
        ["Unregularized correlation", "Graphical-Lasso partial correlation"]
    )
]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]

for score in ["balanced_accuracy", "f1_macro"]:
    for clas in classify:
        df_ = df[df["classes"] == clas]
        df_.reset_index(inplace=True, drop=True)

        ax_score = sns.barplot(
            y="connectivity",
            x=score,
            data=df_,
            orient="h",
            palette=sns.color_palette("coolwarm", n_colors=3),
            hue="lowpass",
            hue_order=[0.1, "Original (0.2)", "Nyquist limit (0.249)"],
            order=[
                "Unregularized correlation",
                "Graphical-Lasso partial correlation",
            ],
        )
        for i in ax_score.containers:
            plt.bar_label(
                i,
                fmt="%.1f",
                label_type="edge",
                fontsize="x-small",
                padding=-60,
                weight="bold",
                color="black",
            )
        wrap_labels(ax_score, 20)
        sns.barplot(
            y="connectivity",
            x=f"dummy_{score}_mostfreq",
            data=df_,
            orient="h",
            hue="lowpass",
            err_kws={"ls": ":"},
            facecolor=(0.8, 0.8, 0.8, 1),
            order=[
                "Unregularized correlation",
                "Graphical-Lasso partial correlation",
            ],
        )
        if score == "balanced_accuracy":
            plt.xlabel("Accuracy")
        else:
            plt.xlabel("F1 score")
        plt.ylabel("FC measure")
        fig = plt.gcf()
        fig.set_size_inches(6, 2.5)
        legend = plt.legend(
            framealpha=0,
            loc="center left",
            bbox_to_anchor=(1, 0.4),
        )
        # remove legend repetition for chance level
        for i, (text, handle) in enumerate(
            zip(legend.texts, legend.legend_handles)
        ):
            if i > 3:
                text.set_visible(False)
                handle.set_visible(False)

            if i == 3:
                text.set_text("Chance-level")
                # set the color of the chance-level label
                handle.set_color((0.8, 0.8, 0.8, 1))

        legend.set_title("Lowpass filter\nfrequency (Hz)")
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{score}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{score}.svg"),
            bbox_inches="tight",
        )
        plt.close("all")
