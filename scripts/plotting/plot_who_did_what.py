"""Image for who-did-what"""

import glob
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ibc_public.utils_data import CONTRASTS, DERIVATIVES, SUBJECTS

sns.set_context("notebook")


def who_did_what(tasks, SUBJECTS, DERIVATIVES):
    they_did_these = pd.DataFrame(index=tasks, columns=SUBJECTS)
    for task in tasks:
        for sub in SUBJECTS:
            # check if there's data for this subject-task pair
            wildcard = os.path.join(
                DERIVATIVES, sub, "*", "func", "*task-%s_*bold.nii.gz" % task
            )
            imgs_ = glob.glob(wildcard)
            if len(imgs_) > 0:
                they_did_these.loc[task, sub] = 1
            else:
                they_did_these.loc[task, sub] = 0
    return they_did_these


def paint_table(they_did_these, output_dir=None, figsize=(8, 16)):
    cmap = sns.color_palette([(1, 1, 1), (0, 0, 0)])
    plt.figure(figsize=figsize)
    sns.heatmap(
        they_did_these.astype(int),
        annot=False,
        cmap=cmap,
        cbar=True,
        linewidths=0.5,
        linecolor="black",
        clip_on=False,
    )

    plt.title("Tasks Completed by Subjects")
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(
            os.path.join(output_dir, "who_did_what.png"), bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output_dir, "who_did_what.svg"),
            bbox_inches="tight",
            transparent=True,
        )
    else:
        plt.savefig("who_did_what.png", dpi=400)
    plt.close()


plots_root = (
    "/data/parietal/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs/"
)
out_dir_name = "who_did_what"
output_dir = os.path.join(plots_root, out_dir_name)
os.makedirs(output_dir, exist_ok=True)

tasks = [
    "RestingState",
    "LePetitPrince",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]
they_did_these = who_did_what(tasks, SUBJECTS, DERIVATIVES)

cmap = sns.color_palette([(1, 1, 1), (0, 0, 0)])
fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(
    they_did_these.astype(int),
    annot=False,
    cmap=cmap,
    cbar=True,
    linewidths=0.5,
    linecolor="black",
    clip_on=False,
    ax=ax,
)
plt.title("Tasks done/not done by each subject", fontweight="bold")
plt.xticks(rotation=40, ha="right")
plt.yticks(rotation=0)
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(["Not Done", "Done"])
colorbar.orientation = "horizontal"
colorbar.outline.set_color("k")
colorbar.outline.set_linewidth(0.5)

plt.savefig(os.path.join(output_dir, "who_did_what.png"), bbox_inches="tight")
plt.savefig(
    os.path.join(output_dir, "who_did_what.svg"),
    bbox_inches="tight",
    transparent=True,
)
plt.close()
