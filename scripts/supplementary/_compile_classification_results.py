import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from glob import glob

results_root = "/Users/himanshu/Desktop/ibc/connectivity/results"

dir_names = [
    "fc_withintask_classification_400_lowpass-nyquist_20250806-182411"
]
for dir_name in dir_names:
    results_pkl = os.path.join(results_root, dir_name, "all_results.pkl")
    pkls = glob(os.path.join(results_root, dir_name, "*", "*.pkl"))
    dfs = []
    for pkl in pkls:
        df = pd.read_pickle(pkl)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.reset_index(inplace=True, drop=True)
    df.to_pickle(results_pkl)
    print(results_pkl)
