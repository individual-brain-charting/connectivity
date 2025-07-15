import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from glob import glob

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"

dir_name = f"classification*"
results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")
pkls = glob(results_pkl)

for pkl in pkls:
    for averaging in ["macro", "micro", "weighted"]:
        df = pd.read_pickle(pkl)
        df.reset_index(inplace=True, drop=True)
        df[f"f1_{averaging}"] = df.apply(
            lambda x: f1_score(
                x["true_class"],
                x["LinearSVC_predicted_class"],
                average=averaging,
            ),
            axis=1,
        )
        df[f"dummy_f1_{averaging}"] = df.apply(
            lambda x: f1_score(
                x["true_class"], x["Dummy_predicted_class"], average=averaging
            ),
            axis=1,
        )
        df.to_pickle(pkl)
        print(pkl)
