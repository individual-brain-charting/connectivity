import os
import sys
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from itertools import compress
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.fetching import get_ses_modality, get_confounds, get_niftis

# fc results directory
fc_results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
tasktype = "natural"

for classification_type in ["across", "within"]:
    for n_parcels in [400, 200]:
        for trim_length in [293, None]:
            original_dir = f"classification-{classification_type}_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}"
            new_dummies_dir = f"classification-{classification_type}_tasktype-{tasktype}_nparcels-{n_parcels}_trim-{trim_length}_dummy_mostfreq"
            print(f"Original: {original_dir}")
            original_path = os.path.join(
                fc_results_root, original_dir, "all_results.pkl"
            )
            new_dummies_path = os.path.join(
                fc_results_root, new_dummies_dir, "all_results.pkl"
            )
            original = pd.read_pickle(original_path)
            new_dummies = pd.read_pickle(new_dummies_path)
            if trim_length == 293:
                new_dummies = new_dummies[new_dummies["classes"] == "Runs"]
            original.reset_index(inplace=True, drop=True)
            new_dummies.reset_index(inplace=True, drop=True)
            for col in list(original.columns):
                if col == "dummy_f1_score":
                    continue
                elif "dummy" in col:
                    print(col)
                    new_col = f"{col}_mostfreq"
                    original[new_col] = new_dummies[col]
                    print(original[new_col].equals(new_dummies[col]))
                else:
                    print(col)
                    try:
                        print(original[col].equals(new_dummies[col]))
                    except KeyError:
                        print(f"KeyError, {col} not in new_dummies")

            original.to_pickle(original_path)
            print(f"Saved {original_path}")
