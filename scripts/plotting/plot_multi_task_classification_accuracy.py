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

results_root = "/storage/store3/work/haggarwa/connectivity/results"
plots_root = "/storage/store3/work/haggarwa/connectivity/plots"
n_parcels = 400
trim_length = None
tasktype = "natural"
classification = "across"
do_hatch = False

dir_name = f"classification-{classification}_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}"
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

df[df.select_dtypes(include=["number"]).columns] *= 100

for clas in classify:
    df_ = df[df["classes"] == clas]
    df_.reset_index(inplace=True, drop=True)

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
        ax_score = sns.barplot(
            y="connectivity",
            x="balanced_accuracy",
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
            x="dummy_balanced_accuracy",
            data=df_,
            orient="h",
            palette=sns.color_palette("pastel")[0:1],
            order=order,
            ci=None,
            facecolor=(0.8, 0.8, 0.8, 1),
        )
        plt.xlabel("Accuracy")
        plt.ylabel("FC measure")
        fig = plt.gcf()
        if how_many == "three":
            fig.set_size_inches(6, 2.5)
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.svg"),
            bbox_inches="tight",
        )
        plt.close("all")
