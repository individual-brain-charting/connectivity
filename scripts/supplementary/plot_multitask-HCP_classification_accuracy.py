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
output_dir = os.path.join(plots_root, "fc_acrosstask_classification_200_HCP")
os.makedirs(output_dir, exist_ok=True)

dir_name = "classification_across-HCP900_tasktype-domain_n_parcels-200_trim-None_20250813-140605"
results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")
df = pd.read_pickle(results_pkl)

# what to classify
classify = ["Subjects"]

df[df.select_dtypes(include=["number"]).columns] *= 100
df = df.rename(
    columns=lambda x: x + "_mostfreq" if x.startswith("dummy_") else x
)
for score in ["balanced_accuracy", "f1_macro"]:
    for clas in classify:
        df_ = df[df["classes"] == clas]
        df_.reset_index(inplace=True, drop=True)

        for how_many in ["all", "three"]:
            if how_many == "all":
                order = [
                    "Unregularized correlation",
                    "Graphical-Lasso partial correlation",
                ]
            elif how_many == "three":
                order = [
                    "Unregularized correlation",
                    "Graphical-Lasso partial correlation",
                ]
            ax_score = sns.barplot(
                y="connectivity",
                x=score,
                data=df_,
                orient="h",
                palette=sns.color_palette()[0:1],
                order=order,
                facecolor=(0.4, 0.4, 0.4, 1),
            )
            for i in ax_score.containers:
                plt.bar_label(
                    i,
                    fmt="%.1f",
                    label_type="edge",
                    fontsize="x-small",
                    padding=-45,
                    weight="bold",
                    color="white",
                )
            wrap_labels(ax_score, 20)
            sns.barplot(
                y="connectivity",
                x=f"dummy_{score}_mostfreq",
                data=df_,
                orient="h",
                palette=sns.color_palette("pastel")[0:1],
                order=order,
                facecolor=(0.8, 0.8, 0.8, 1),
                err_kws={"ls": ":"},
            )
            if score == "balanced_accuracy":
                plt.xlabel("Accuracy")
            else:
                plt.xlabel("F1 score")
            plt.ylabel("FC measure")
            fig = plt.gcf()
            if how_many == "three":
                fig.set_size_inches(6, 2.5)
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
            )
            plt.close("all")
