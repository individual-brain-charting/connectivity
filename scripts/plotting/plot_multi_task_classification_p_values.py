import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.plot import wrap_labels, compute_corrected_ttest
from scipy.stats import wilcoxon, ttest_rel

root = "/Users/himanshu/Desktop/ibc/connectivity"
# root = "/data/parietal/store3/work/haggarwa/connectivity"
results_root = "results/wo_extra_GBU_runs"

n_parcels = 400
trim_length = None
tasktype = "natural"
do_hatch = False

dir_name = f"classification-across_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}"
results_pkl = os.path.join(root, results_root, dir_name, "all_results.pkl")

df = pd.read_pickle(results_pkl)

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]

df[df.select_dtypes(include=["number"]).columns] *= 100

p_values = {}
for score in ["balanced_accuracy", "f1_macro"]:
    p_values[score] = {}
    for clas in classify:
        p_values[score][clas] = {}
        df_ = df[df["classes"] == clas]
        df_.reset_index(inplace=True, drop=True)
        for between in [
            "Unregularized correlation vs. Graphical-Lasso partial correlation",
            "Ledoit-Wolf correlation vs. Graphical-Lasso partial correlation",
        ]:
            p_values[score][clas][between] = {}
            scores_1 = df_[df_["connectivity"] == between.split(" vs. ")[0]][
                score
            ]
            scores_2 = df_[df_["connectivity"] == between.split(" vs. ")[1]][
                score
            ]
            if clas in ["Runs", "Tasks"]:
                alt = "greater"
            else:
                alt = "less"
            p_values[score][clas][between]["wilcoxon"] = wilcoxon(
                scores_1, scores_2, alternative=alt
            )[1]
            p_values[score][clas][between]["ttest_rel"] = ttest_rel(
                scores_1, scores_2, alternative=alt
            )[1]

            # scores of the best model
            model_1_scores = scores_1.values
            # scores of the second-best model
            model_2_scores = scores_2.values
            if alt == "greater":
                differences = model_1_scores - model_2_scores
            else:
                differences = model_2_scores - model_1_scores

            n = differences.shape[0]  # number of test sets
            dof = n - 1
            n_train = df_["train_sets"].loc[0].shape[0]
            n_test = df_["test_sets"].loc[0].shape[0]

            p_values[score][clas][between]["corrected_ttest"] = (
                compute_corrected_ttest(differences, dof, n_train, n_test)[1]
            )
            print(
                f"{between} {clas} {score}: "
                f"wilcoxon: {p_values[score][clas][between]['wilcoxon']}, "
                f"ttest_rel: {p_values[score][clas][between]['ttest_rel']}, "
                f"corrected_ttest: {p_values[score][clas][between]['corrected_ttest']}"
            )
