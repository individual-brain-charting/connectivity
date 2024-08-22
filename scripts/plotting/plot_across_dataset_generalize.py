import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob
from utils.plot import wrap_labels

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")

results = "/storage/store3/work/haggarwa/connectivity/results/across_dataset_generalize"
plots = "/storage/store3/work/haggarwa/connectivity/plots/"
output = os.path.join(plots, "across_dataset_generalize")
os.makedirs(output, exist_ok=True)

data_files = glob(os.path.join(results, "*.csv"))
for data_file in data_files:
    # split path to get file name
    filename = data_file.split("/")[-1]
    # split file name to get dataset names
    dataset = filename.split("_")[-2].split("-")[1]
    for score in ["balanced_accuracy", "f1"]:
        data = pd.read_csv(
            data_file,
        )
        data[data.select_dtypes(include=["number"]).columns] *= 100
        datasets = data["direction"].str.split(" -> ", expand=True)[0].unique()
        order = [
            "Unregularized correlation",
            "Ledoit-Wolf correlation",
            "Graphical-Lasso partial correlation",
        ]
        ax_score = sns.barplot(
            y="cov measure",
            x=score,
            hue="direction",
            data=data,
            orient="h",
            palette="Dark2",
            order=order,
        )
        wrap_labels(ax_score, 20)
        ax_dummy = sns.barplot(
            y="cov measure",
            x=f"dummy_{score}",
            hue="direction",
            data=data,
            orient="h",
            palette="Pastel2",
            order=order,
        )
        if score == "balanced_accuracy":
            plt.xlabel("Accuracy")
        else:
            plt.xlabel("F1 score")
        plt.ylabel("FC measure")
        plt.title(
            f"Generalization across datasets: {datasets[0]}, {datasets[1]}"
        )
        plt.legend(framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5))
        plot_file = data_file.split("/")[-1].split(".")[0]
        plt.savefig(
            os.path.join(output, f"{plot_file}_{score}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output, f"{plot_file}_{score}.svg"),
            bbox_inches="tight",
        )
        plt.close()
