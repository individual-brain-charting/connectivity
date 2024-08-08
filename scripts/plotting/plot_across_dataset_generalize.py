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
    data = pd.read_csv(data_file)
    datasets = data["direction"].str.split(" -> ", expand=True)[0].unique()
    order = [
        "Unregularized correlation",
        "Ledoit-Wolf correlation",
        "Graphical-Lasso partial correlation",
    ]
    ax_score = sns.barplot(
        y="cov measure",
        x="balanced_accuracy",
        hue="direction",
        data=data,
        orient="h",
        palette="tab10",
        order=order,
    )
    wrap_labels(ax_score, 20)
    ax_dummy = sns.barplot(
        y="cov measure",
        x="dummy_balanced_accuracy",
        hue="direction",
        data=data,
        orient="h",
        palette="pastel",
        order=order,
    )
    plt.xlabel("Balanced Accuracy")
    plt.ylabel("FC measure")
    plt.title(f"Generalization across datasets: {datasets[0]}, {datasets[1]}")
    plt.legend(framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5))
    plot_file = data_file.split("/")[-1].split(".")[0]
    plt.savefig(os.path.join(output, f"{plot_file}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output, f"{plot_file}.svg"), bbox_inches="tight")
    plt.close()
