import os
import pandas as pd
import seaborn as sns

### table of all accuracies ###

n_parcels = 200
trim_length = 293
tasktype = "natural"

plots_root = f"/data/parietal/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs/meanscores_nparcels-{n_parcels}_trim-{trim_length}"

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"

within_dir = f"classification-within_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}"
within_pkl = os.path.join(results_root, within_dir, "all_results.pkl")
across_dir = f"classification-across_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}"
across_pkl = os.path.join(results_root, across_dir, "all_results.pkl")

across_df = pd.read_pickle(across_pkl).reset_index(drop=True)
within_df = pd.read_pickle(within_pkl).reset_index(drop=True)

output_dir = os.path.join(plots_root)
os.makedirs(output_dir, exist_ok=True)


# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
if trim_length is None:
    classify = ["Tasks", "Subjects", "Runs"]
elif trim_length == 293:
    classify = ["Runs"]
else:
    raise ValueError("trim_length not recognized")
# tasks
tasks = [
    "RestingState",
    "LePetitPrince",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]

# get accuracies for each classification scenario
for clas in classify:
    print(clas)
    classifying_df = pd.concat(
        [
            across_df[across_df["classes"] == clas],
            within_df[within_df["classes"] == clas],
        ]
    )
    classifying_df.reset_index(inplace=True, drop=True)
    for metric in [
        "balanced_accuracy",
        "dummy_balanced_accuracy_mostfreq",
        "f1_macro",
        "dummy_f1_macro_mostfreq",
    ]:
        mean_acc = (
            classifying_df.groupby(["task_label", "connectivity"])[metric]
            .mean()
            .round(2)
        )
        mean_acc = mean_acc.unstack(level=1)
        mean_acc["mean"] = mean_acc.mean(axis=1).round(2)
        mean_acc = mean_acc[
            [
                "Unregularized correlation",
                "Unregularized partial correlation",
                "Ledoit-Wolf correlation",
                "Ledoit-Wolf partial correlation",
                "Graphical-Lasso correlation",
                "Graphical-Lasso partial correlation",
                "mean",
            ]
        ]
        mean_acc.to_csv(os.path.join(output_dir, f"{clas}_mean_{metric}.csv"))
