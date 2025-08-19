import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from glob import glob

results_root = "/Users/himanshu/Desktop/ibc/connectivity/results"

dir_names = [
    "classification_across-HCP900_tasktype-domain_n_parcels-200_trim-None_20250813-140605",
]
for dir_name in dir_names:
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
                    x["true_class"],
                    x["Dummy_predicted_class"],
                    average=averaging,
                ),
                axis=1,
            )
            df.to_pickle(pkl)
            print(pkl)
