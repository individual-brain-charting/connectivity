import os
import pandas as pd
from glob import glob
from joblib import dump, load

results_root = "/Users/himanshu/Desktop/ibc/connectivity/results"

dir_names = ["permutation_tests"]
for dir_name in dir_names:
    results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")
    pkls = glob(os.path.join(results_root, dir_name, "*", "*.pkl"))
    dfs = []
    for pkl in pkls:
        print(f"Processing {pkl}...")
        pkl = load(pkl)
        df = pd.DataFrame(pkl)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.reset_index(inplace=True, drop=True)
    mean_permutation_scores = []
    std_permutation_scores = []
    for index, row in df.iterrows():
        mean_permutation_scores.append(row["permutation_scores"].mean())
        std_permutation_scores.append(row["permutation_scores"].std())
    df["mean_permutation_scores"] = mean_permutation_scores
    df["std_permutation_scores"] = std_permutation_scores
    df.to_pickle(results_pkl)
    print(results_pkl)
