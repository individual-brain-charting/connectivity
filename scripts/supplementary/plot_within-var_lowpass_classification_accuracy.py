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
    plots_root, "fc_withintask_classification_400_var-lowpass"
)
os.makedirs(output_dir, exist_ok=True)

dir_name = "fc_withintask_classification_400_lowpass-0.5_20250801-141032"
pkl_05 = os.path.join(results_root, dir_name, "all_results.pkl")

dir_name = "fc_withintask_classification_400_lowpass-0.1_20250801-141038"
pkl_01 = os.path.join(results_root, dir_name, "all_results.pkl")

dir_name = "wo_extra_GBU_runs/classification-within_tasktype-natural_nparcels-400_trim-None"
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

df = df[df["connectivity"] == "Unregularized correlation"]

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

for score in ["balanced_accuracy", "f1_macro"]:
    for clas in classify:
        df_ = df[df["classes"] == clas]
        df_.reset_index(inplace=True, drop=True)
        if clas == "Runs":
            print(len(df_))
            print(df_["task_label"].unique())
            df_ = df_[df_["task_label"].isin(movies)]
            print(len(df_))
        order = [0.1, "Original (0.2)", "Nyquist limit (0.249)"]

        ax_score = sns.barplot(
            y="connectivity",
            x=score,
            data=df_,
            orient="h",
            hue="lowpass",
            palette=sns.color_palette("coolwarm", n_colors=3),
            hue_order=order,
            # errwidth=1,
        )
        wrap_labels(ax_score, 20)
        for i, container in enumerate(ax_score.containers):
            plt.bar_label(
                container,
                fmt="%.1f",
                label_type="edge",
                fontsize="x-small",
                padding=-100,
                weight="bold",
                color="black",
            )
        ax_chance = sns.barplot(
            y="connectivity",
            x="dummy_" + score + "_mostfreq",
            data=df_,
            orient="h",
            hue="lowpass",
            hue_order=order,
            facecolor=(0.8, 0.8, 0.8, 1),
            err_kws={"ls": ":"},
        )
        if score == "balanced_accuracy":
            plt.xlabel("Accuracy")
        else:
            plt.xlabel("F1 score")
        plt.ylabel("FC measure")
        legend = plt.legend(
            framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
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
        fig = plt.gcf()
        fig.set_size_inches(6, 2.5)

        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{score}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{score}.svg"),
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()
