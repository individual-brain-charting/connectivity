"""This script calculates the similarity between functional connectivity
 matrices from different tasks and structural connectivity"""

import os
import time
import pandas as pd
from nilearn import datasets
from joblib import Parallel, delayed

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.similarity import (
    mean_connectivity,
    get_similarity,
)


fc_root = (
    "/storage/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
)
sc_root = "/storage/store3/work/haggarwa/connectivity/results"
n_parcels = 400
trim_length = None

fc_pkl = (
    f"connectomes_nparcels-{n_parcels}_tasktype-natural_trim-{trim_length}.pkl"
)
fc_pkl = os.path.join(fc_root, fc_pkl)
sc_pkl = f"sc_data_native_{n_parcels}"
sc_pkl = os.path.join(sc_root, sc_pkl)

output_dir = f"fcsc_similarity_nparcels-{n_parcels}_trim-{trim_length}"
output_dir = os.path.join(fc_root, output_dir)
os.makedirs(output_dir, exist_ok=True)
# number of jobs to run in parallel
n_jobs = 50
# tasks
tasks = [
    "RestingState",
    "LePetitPrince",
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
    ("RestingState", "LePetitPrince"),
    ("RestingState", "Raiders"),
    ("RestingState", "GoodBadUgly"),
    ("RestingState", "MonkeyKingdom"),
    ("RestingState", "Mario"),
    ("LePetitPrince", "Raiders"),
    ("LePetitPrince", "GoodBadUgly"),
    ("LePetitPrince", "MonkeyKingdom"),
    ("Raiders", "GoodBadUgly"),
    ("Raiders", "MonkeyKingdom"),
    ("GoodBadUgly", "MonkeyKingdom"),
    ("LePetitPrince", "Mario"),
    ("Raiders", "Mario"),
    ("GoodBadUgly", "Mario"),
    ("MonkeyKingdom", "Mario"),
    ("RestingState", "SC"),
    ("LePetitPrince", "SC"),
    ("Raiders", "SC"),
    ("GoodBadUgly", "SC"),
    ("MonkeyKingdom", "SC"),
    ("Mario", "SC"),
    ("SC", "SC"),
]


def all_combinations(task_pairs, cov_estimators, measures):
    """generator to yield all combinations of task pairs, cov estimators, to
    parallelize the similarity calculation for each combination"""
    for task_pair in task_pairs:
        for cov in cov_estimators:
            for measure in measures:
                yield task_pair, cov, measure


data = pd.read_pickle(fc_pkl)
data = data[data["dataset"] == "ibc"]

sc_data = pd.read_pickle(sc_pkl)

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
