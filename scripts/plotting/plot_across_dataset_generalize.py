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

results = "/data/parietal/store3/work/haggarwa/connectivity/results/across_dataset_generalize"
plots = "/data/parietal/store3/work/haggarwa/connectivity/plots/"
output = os.path.join(plots, "across_dataset_generalize")
os.makedirs(output, exist_ok=True)

dataset_name_replacements = {
    "archi": "ARCHI",
    "ibc": "IBC",
    "HCP900": "HCP",
    "thelittleprince": "TheLittlePrince",
    "HumanMonkeyGBU": "HumanMonkeyGBU",
}

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
        # replace "->" with →
        data["direction"] = data["direction"].str.replace("->", "→")
        # replace dataset names using dictionary
        data["direction"] = (
            data["direction"]
            .str.split(" → ", expand=True)[0]
            .map(dataset_name_replacements)
            + " → "
            + data["direction"]
            .str.split(" → ", expand=True)[1]
            .map(dataset_name_replacements)
        )
        data[data.select_dtypes(include=["number"]).columns] *= 100
        datasets = data["direction"].str.split(" → ", expand=True)[0].unique()
        # replace dataset names
        datasets = [
            dataset_name_replacements.get(item, item) for item in datasets
        ]
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
            palette=sns.color_palette("pastel")[7:],
            facecolor=(0.8, 0.8, 0.8, 1),
            order=order,
        )
        if score == "balanced_accuracy":
            plt.xlabel("Accuracy")
        else:
            plt.xlabel("F1 score")
        plt.ylabel("FC measure")
        # plt.title(f"{datasets[0]} and {datasets[1]}")
        legend = plt.legend(
            framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        legend_cutoff = 2
        # remove legend repetition for chance level
        for i, (text, handle) in enumerate(
            zip(legend.texts, legend.legend_handles)
        ):
            if i > legend_cutoff:
                text.set_visible(False)
                handle.set_visible(False)
            if i == legend_cutoff:
                text.set_text("Chance-level")
                handle.set_color((0.8, 0.8, 0.8, 1))
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
