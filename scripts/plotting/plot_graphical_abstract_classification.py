import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot import wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.6)

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
plots_root = (
    "/data/parietal/store3/work/haggarwa/connectivity/plots/graphical_abstract"
)
n_parcels = 400
trim_length = None
tasktype = "natural"
do_hatch = False

dir_name = f"classification-across_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")

output_dir = os.path.join(plots_root, dir_name)
os.makedirs(output_dir, exist_ok=True)
df = pd.read_pickle(results_pkl)

# cov estimators
cov_estimators = ["Graphical-Lasso", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]

df[df.select_dtypes(include=["number"]).columns] *= 100

for score in ["f1_macro"]:
    for clas in classify:
        df_ = df[df["classes"] == clas]
        df_.reset_index(inplace=True, drop=True)
        df_["connectivity"] = df_["connectivity"].map(
            {
                "Unregularized correlation": "Correlation",
                "Graphical-Lasso partial correlation": "Partial correlation",
            }
        )
        order = [
            "Correlation",
            "Partial correlation",
        ]
        ax_score = sns.barplot(
            y="connectivity",
            x=score,
            data=df_,
            orient="h",
            palette=sns.color_palette()[0:1],
            order=order,
        )
        for i in ax_score.containers:
            plt.bar_label(
                i,
                fmt="%.1f",
                label_type="edge",
                fontsize="x-small",
                padding=-75,
                weight="bold",
                color="white",
            )
        wrap_labels(ax_score, 10)
        sns.barplot(
            y="connectivity",
            x=f"dummy_{score}_mostfreq",
            data=df_,
            orient="h",
            palette=sns.color_palette("pastel")[0:1],
            order=order,
            err_kws={"ls": ":"},
        )
        if score == "balanced_accuracy":
            plt.xlabel("Accuracy")
        else:
            plt.xlabel("F1 score")
        plt.ylabel("")
        # create custom legend
        # replace lines with patches of color for legend
        handles = [
            Patch(color=sns.color_palette()[0], label="LinearSVC"),
            Patch(
                color=sns.color_palette("pastel")[0], label="DummyClassifier"
            ),
        ]
        ax_score.legend(
            handles=handles,
            framealpha=0,
            bbox_to_anchor=(1, 0.5),
            loc="center left",
            title="Classifier",
        )
        fig = plt.gcf()
        fig.set_size_inches(6, 2.5)
        if clas == "Tasks":
            plt.title("Task classification")
        elif clas == "Subjects":
            plt.title("Subject fingerprinting")
        elif clas == "Runs":
            plt.title("Run classification")
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{score}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{score}.svg"),
            bbox_inches="tight",
        )
        plt.close("all")
