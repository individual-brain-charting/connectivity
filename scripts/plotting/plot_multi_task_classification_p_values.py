import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.plot import wrap_labels
from scipy.stats import wilcoxon, ttest_rel

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
plots_root = (
    "/data/parietal/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs"
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
