import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from glob import glob

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results"
output_dir = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
os.makedirs(output_dir, exist_ok=True)

wildcard = f"connectomes*natural*.pkl"
pkls = glob(os.path.join(results_root, wildcard))

for pkl in pkls:
    df = pd.read_pickle(pkl)
    # create new colum with run number
    df["run_nums"] = df["run_labels"].str.extract(r"(\d+)")
    # convert run_nums to int
    df["run_nums"] = pd.to_numeric(df["run_nums"], errors="coerce")
    # find indices where task is GoodBadUgly and run_labels are not b/w
    # run-03 and run-17
    indices = df[
        (df["dataset"] == "ibc")
        & (df["tasks"] == "GoodBadUgly")
        & ((df["run_nums"] < 3) | (df["run_nums"] > 17))
    ].index

    # drop rows with indices
    df.drop(indices, inplace=True)
    df.drop(columns="run_nums", inplace=True)
    df.reset_index(inplace=True, drop=True)
    try:
        assert len(indices) == 66
    except AssertionError:
        print("AssertionError")
        print(pkl)
        print(before_cnt, after_cnt)
        print(len(indices))
    to_save = os.path.join(output_dir, pkl.split("/")[-1])
    df.to_pickle(to_save)
    print(pkl, to_save)
