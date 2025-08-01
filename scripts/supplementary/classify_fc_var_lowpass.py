"""Perform FC classification over runs, subjects and tasks."""

import os
import time

import pandas as pd
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed
import sys

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.fc_classification import do_cross_validation

# read arguments
if len(sys.argv) > 1:
    low_pass = float(sys.argv[1])
    within_task = True if sys.argv[2] == "within" else False
else:
    low_pass = 0.2
    within_task = False

#### INPUTS
# number of jobs to run in parallel
n_jobs = 20
# number of parcels
n_parcels = 400  # or 400
# number of splits for cross validation
n_splits = 50
# trim to use
trim = None
# we will use the resting state and all the movie-watching sessions
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
    "LePetitPrince",
]
# cov estimators
cov_estimators = [
    # "Graphical-Lasso",
    # "Ledoit-Wolf",
    "Unregularized",
]
# connectivity measures for each cov estimator
measures = [
    "correlation",
    # "partial correlation",
]
# what to classify
classify = ["Runs", "Subjects", "Tasks"]

#### SETUP
# concatenate estimator and measure names
connectivity_measures = []
for cov in cov_estimators:
    for measure in measures:
        connectivity_measures.append(cov + " " + measure)

# cache and root output directory
data_root = "/data/parietal/store3/work/haggarwa/connectivity/data/"
# results directory
results = "/data/parietal/store3/work/haggarwa/connectivity/results/"
os.makedirs(results, exist_ok=True)

if within_task:
    output_dir = (
        f"fc_withintask_classification_{n_parcels}_lowpass-{low_pass}"
        f"_{time.strftime('%Y%m%d-%H%M%S')}"
    )
else:
    output_dir = (
        f"fc_acrosstask_classification_{n_parcels}_lowpass-{low_pass}"
        f"_{time.strftime('%Y%m%d-%H%M%S')}"
    )
output_dir = os.path.join(results, output_dir)
os.makedirs(output_dir, exist_ok=True)

# connectivity data path
fc_data_path = os.path.join(
    results,
    f"connectomes_nparcels-{n_parcels}_tasktype-natural_trim-{trim}_lowpass-{low_pass}.pkl",
)


# generator to get all the combinations of classification
# for later running all cases in parallel
def all_combinations(classify, tasks, connectivity_measures, within_task):
    # dictionary to map the classification to the tasks
    ## within each task
    # when classifying by runs or subjects, we classify runs or subjects
    #  within each task when classifying by tasks, we classify between
    # two tasks in this case, RestingState vs. each movie-watching task
    if within_task:
        tasks_ = {
            "Runs": tasks,
            "Subjects": tasks,
            "Tasks": [
                ["RestingState", "Raiders"],
                ["RestingState", "GoodBadUgly"],
                ["RestingState", "MonkeyKingdom"],
                ["RestingState", "LePetitPrince"],
                ["RestingState", "Mario"],
                ["Raiders", "GoodBadUgly"],
                ["Raiders", "MonkeyKingdom"],
                ["GoodBadUgly", "MonkeyKingdom"],
                ["LePetitPrince", "Raiders"],
                ["LePetitPrince", "GoodBadUgly"],
                ["LePetitPrince", "MonkeyKingdom"],
                ["LePetitPrince", "Mario"],
                ["Raiders", "Mario"],
                ["GoodBadUgly", "Mario"],
                ["MonkeyKingdom", "Mario"],
            ],
        }
    ## across all tasks
    else:
        tasks_ = {
            # only classify runs across movie-watching tasks
            "Runs": [
                [
                    "Raiders",
                    "GoodBadUgly",
                    "MonkeyKingdom",
                    "LePetitPrince",
                ]
            ],
            "Subjects": [tasks],
            "Tasks": [tasks],
        }
    for classes in classify:
        for task in tasks_[classes]:
            for connectivity_measure in connectivity_measures:
                yield classes, task, connectivity_measure


#### LOAD CONNECTIVITY
data = pd.read_pickle(fc_data_path)

#### RUN CLASSIFICATION
# run classification for all combinations of classification, task and
# connectivity measure in parallel
# do_cross_validation returns a dataframe with the results of the cross
# validation for each case
if within_task:
    print("Starting within task cross validation......")
else:
    print("Starting across task cross validation......")
all_results = Parallel(n_jobs=n_jobs, verbose=11, backend="loky")(
    delayed(do_cross_validation)(
        classes, task, n_splits, connectivity_measure, data, output_dir
    )
    for classes, task, connectivity_measure in all_combinations(
        classify, tasks, connectivity_measures, within_task
    )
)

#### SAVE RESULTS
print("Saving results...")
all_results = pd.concat(all_results)
# save the results
all_results.to_pickle(os.path.join(output_dir, "all_results.pkl"))
