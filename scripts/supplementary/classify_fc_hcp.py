"""Perform FC classification over runs, subjects and tasks."""

import os
import time
import sys

import pandas as pd
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.fc_classification import do_cross_validation

# read arguments
if len(sys.argv) > 1:
    low_pass = sys.argv[1]
    within_task = True if sys.argv[2] == "within" else False
else:
    low_pass = 0.2
    within_task = False

#### INPUTS
# number of jobs to run in parallel
n_jobs = 40
# number of parcels
n_parcels = 200  # or 400
# number of splits for cross validation
n_splits = 50
# trim to use
trim = None
tasks = [
    "EMOTION",
    "GAMBLING",
    "LANGUAGE",
    "MOTOR",
    "RELATIONAL",
    "SOCIAL",
    "WM",
]
# what to classify
classify = ["Subjects"]

#### SETUP
# concatenated estimator and measure names
connectivity_measures = [
    "Graphical-Lasso partial correlation",
    "Unregularized correlation",
]

# cache and root output directory
data_root = "/data/parietal/store3/work/haggarwa/connectivity/data/"
# results directory
results = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
os.makedirs(results, exist_ok=True)

if within_task:
    output_dir = (
        f"classification_within-HCP900_tasktype-domain"
        f"_n_parcels-{n_parcels}_trim-{trim}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
else:
    output_dir = (
        f"classification_across-HCP900_tasktype-domain"
        f"_n_parcels-{n_parcels}_trim-{trim}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
output_dir = os.path.join(results, output_dir)
os.makedirs(output_dir, exist_ok=True)

# connectivity data path
fc_data_path = os.path.join(
    results,
    f"connectomes_nparcels-{n_parcels}_tasktype-domain_trim-{trim}.pkl",
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
data = data[data["dataset"] == "HCP900"]
data.reset_index(drop=True, inplace=True)

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
