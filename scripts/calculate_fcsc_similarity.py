"""This script calculates the similarity between functional connectivity
 matrices from different tasks and structural connectivity"""

import os
import time
import pandas as pd
from nilearn import datasets
from joblib import Parallel, delayed

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.similarity import (
    mean_connectivity,
    get_similarity,
)
from utils.fc_estimation import (
    get_connectomes,
    get_time_series,
)

cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
output_dir = f"fc_similarity_{time.strftime('%Y%m%d-%H%M%S')}"
output_dir = os.path.join(DATA_ROOT, output_dir)
os.makedirs(output_dir, exist_ok=True)
calculate_connectivity = False
n_parcels = 400
if n_parcels == 400:
    fc_data_path = os.path.join(cache, "connectomes_400_comprcorr")
    sc_data_path = os.path.join(cache, "sc_data_native_new")
elif n_parcels == 200:
    fc_data_path = os.path.join(cache, "connectomes_200_comprcorr")
    sc_data_path = os.path.join(cache, "sc_data_native_200")
# number of jobs to run in parallel
n_jobs = 50
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]

task_pairs = [
    ("RestingState", "Raiders"),
    ("RestingState", "GoodBadUgly"),
    ("RestingState", "MonkeyKingdom"),
    ("RestingState", "Mario"),
    ("Raiders", "GoodBadUgly"),
    ("Raiders", "MonkeyKingdom"),
    ("GoodBadUgly", "MonkeyKingdom"),
    ("Raiders", "Mario"),
    ("GoodBadUgly", "Mario"),
    ("MonkeyKingdom", "Mario"),
    ("RestingState", "SC"),
    ("Raiders", "SC"),
    ("GoodBadUgly", "SC"),
    ("MonkeyKingdom", "SC"),
    ("Mario", "SC"),
]


def all_combinations(task_pairs, cov_estimators, measures):
    """generator to yield all combinations of task pairs, cov estimators, to
    parallelize the similarity calculation for each combination"""
    for task_pair in task_pairs:
        for cov in cov_estimators:
            for measure in measures:
                yield task_pair, cov, measure


data = pd.read_pickle(fc_data_path)
sc_data = pd.read_pickle(sc_data_path)

all_connectivity = mean_connectivity(data, tasks, cov_estimators, measures)
all_connectivity = pd.concat([all_connectivity, sc_data], axis=0)

results = Parallel(n_jobs=n_jobs, verbose=2, backend="loky")(
    delayed(get_similarity)(all_connectivity, task_pair, cov, measure)
    for task_pair, cov, measure in all_combinations(
        task_pairs, cov_estimators, measures
    )
)

results = [item for sublist in results for item in sublist]
results = pd.DataFrame(results)
results.to_pickle(os.path.join(output_dir, "results.pkl"))