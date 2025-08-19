import os
import pandas as pd
from glob import glob
from joblib import dump, load

results_root = "/Users/himanshu/Desktop/ibc/connectivity/results"

dir_names = ["hcp_subject_fingerprinting_pairwise_tasks"]
for dir_name in dir_names:
    results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")
    pkls = glob(os.path.join(results_root, dir_name, "*.pkl"))
    dfs = []
    for pkl in pkls:
        print(f"Processing {pkl}...")
        pkl = load(pkl)
        pkl["f1_scores"] = pkl["scores"]["test_f1_macro"]
        pkl["balanced_accuracies"] = pkl["scores"]["test_balanced_accuracy"]
        # remove "scores" key from the dictionary
        pkl["scores"] = 1
        df = pd.DataFrame(pkl)
        df.drop(columns=["scores"], inplace=True)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.reset_index(inplace=True, drop=True)
    df.to_pickle(results_pkl)
    print(results_pkl)
