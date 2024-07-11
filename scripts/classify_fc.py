"""Perform FC classification over runs, subjects and tasks."""

import os
import time

import pandas as pd
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.fc_estimation import (
    get_connectomes,
    get_time_series,
)
from utils.fc_classification import do_cross_validation

sns.set_theme(context="talk", style="whitegrid")

#### INPUTS
# number of jobs to run in parallel
n_jobs = 1
# number of parcels
n_parcels = 400  # or 400
# number of splits for cross validation
n_splits = 50
# do within each task or across all tasks
within_task = True
# we will use the resting state and all the movie-watching sessions
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Runs", "Subjects", "Tasks"]

#### SETUP
# concatenate estimator and measure names
connectivity_measures = []
for cov in cov_estimators:
    for measure in measures:
        connectivity_measures.append(cov + " " + measure)

# cache and root output directory
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
if within_task:
    output_dir = (
        f"fc_withintask_classification_{n_parcels}"
        f"_{time.strftime('%Y%m%d-%H%M%S')}"
    )
else:
    output_dir = (
        f"fc_acrosstask_classification_{n_parcels}_"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )
output_dir = os.path.join(DATA_ROOT, output_dir)
os.makedirs(output_dir, exist_ok=True)

# connectivity data path
if n_parcels == 400:
    # with compcorr
    fc_data_path = os.path.join(cache, "connectomes_400_comprcorr")
elif n_parcels == 200:
    # with compcorr
    fc_data_path = os.path.join(cache, "connectomes_200_comprcorr")


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
                ["RestingState", "Mario"],
                ["Raiders", "GoodBadUgly"],
                ["Raiders", "MonkeyKingdom"],
                ["GoodBadUgly", "MonkeyKingdom"],
                ["Raiders", "Mario"],
                ["GoodBadUgly", "Mario"],
                ["MonkeyKingdom", "Mario"],
            ],
        }
    ## across all tasks
    else:
        tasks_ = {
            # only classify runs across movie-watching tasks
            "Runs": [["Raiders", "GoodBadUgly", "MonkeyKingdom"]],
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
all_results = Parallel(n_jobs=10, verbose=11, backend="loky")(
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