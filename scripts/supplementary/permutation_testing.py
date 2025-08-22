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
from utils.fc_classification import do_permute_test

# read arguments
if len(sys.argv) > 1:
    low_pass = sys.argv[1]
    within_task = True if sys.argv[2] == "within" else False
    classify = [sys.argv[3]]
    connectivity_measures = [sys.argv[4]]
else:
    low_pass = 0.2
    within_task = False
    classify = ["Subjects", "Runs", "Tasks"]
    connectivity_measures = [
        "Graphical-Lasso partial correlation",
        "Unregularized correlation",
        "Ledoit-Wolf correlation",
    ]

#### INPUTS
# number of jobs to run in parallel
n_jobs = n_permutations = 50
# number of parcels
n_parcels = 400
# trim to use
trim = None
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
    "LePetitPrince",
]

#### SETUP

# cache and root output directory
data_root = "/data/parietal/store3/work/haggarwa/connectivity/data/"
# results directory
results = "/data/parietal/store3/work/haggarwa/connectivity/results/wo_extra_GBU_runs"
os.makedirs(results, exist_ok=True)

if within_task:
    output_dir = (
        f"classification_within-permute_tasktype-natural"
        f"_n_parcels-{n_parcels}_trim-{trim}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
else:
    output_dir = (
        f"classification_across-permute_tasktype-natural"
        f"_n_parcels-{n_parcels}_trim-{trim}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
output_dir = os.path.join(results, "permutation_tests", output_dir)
os.makedirs(output_dir, exist_ok=True)

# connectivity data path
fc_data_path = os.path.join(
    results,
    f"connectomes_nparcels-{n_parcels}_tasktype-natural_trim-{trim}.pkl",
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
data = data[data["dataset"] == "ibc"]
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

all_results = []
for classes, task, connectivity_measure in all_combinations(
    classify, tasks, connectivity_measures, within_task
):
    all_results.append(
        do_permute_test(
            classes,
            task,
            n_splits,
            connectivity_measure,
            data,
            output_dir,
            n_permutations,
        )
    )

#### SAVE RESULTS
print("Saving results...")
all_results = pd.concat(all_results)
# save the results
all_results.to_pickle(os.path.join(output_dir, "all_results.pkl"))
