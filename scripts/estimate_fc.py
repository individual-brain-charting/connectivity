"""Pipeline to estimate functional connectivity matrices"""

import os
import sys
import pandas as pd
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.fc_estimation import (
    get_connectomes,
    get_time_series,
)

#### INPUTS
# kind of tasks to keep
#  - "natural" for naturalistic tasks
#  - "domain" for tasks from different domains
tasktype = "domain"
# trim the time series to the given length, None to keep all
# keeping 293 time points for natural tasks
# 128 for domain tasks
trim_length = 128 if tasktype == "domain" else 293
# cache and root output directory
data_root = "/storage/store3/work/haggarwa/connectivity/data/"
# results directory
results = "/storage/store3/work/haggarwa/connectivity/results"
os.makedirs(results, exist_ok=True)
# number of jobs to run in parallel
n_jobs = 10
# number of parcels: 400 or 200
n_parcels = 200
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# datasets and tasks to extract time series from
dataset_task = {
    "ibc": [
        # Naturalistic
        "GoodBadUgly",
        "MonkeyKingdom",
        "Raiders",
        "RestingState",
        "Mario",
        "LePetitPrince",
        # Archi
        "ArchiStandard",
        "ArchiSpatial",
        "ArchiSocial",
        "ArchiEmotional",
        # HCP
        "HcpEmotion",
        "HcpGambling",
        "HcpLanguage",
        "HcpMotor",
        "HcpRelational",
        "HcpSocial",
        "HcpWm",
    ],
    "HCP900": [
        "EMOTION",
        "GAMBLING",
        "LANGUAGE",
        "MOTOR",
        "RELATIONAL",
        "SOCIAL",
        "WM",
    ],
    "archi": [
        "ArchiStandard",
        "ArchiSpatial",
        "ArchiSocial",
        "ArchiEmotional",
    ],
    "thelittleprince": ["lppFR"],
}


#### SETUP
# output data paths
timeseries_path = os.path.join(
    results,
    f"timeseries_nparcels-{n_parcels}_tasktype-{tasktype}_trim-{trim_length}.pkl",
)
fc_data_path = os.path.join(
    results,
    f"connectomes_nparcels-{n_parcels}_tasktype-{tasktype}_trim-{trim_length}.pkl",
)


# filter tasks and generate dataset-task pairs
def generator_dataset_task(dataset_task, tasktype):
    if tasktype == "natural":
        dataset_task.pop("archi", None)
        dataset_task.pop("HCP900", None)
        dataset_task["ibc"] = [
            "GoodBadUgly",
            "MonkeyKingdom",
            "Raiders",
            "Mario",
            "LePetitPrince",
            "RestingState",
        ]
    elif tasktype == "domain":
        dataset_task.pop("thelittleprince", None)
        dataset_task["ibc"] = [
            "ArchiStandard",
            "ArchiSpatial",
            "ArchiSocial",
            "ArchiEmotional",
            "HcpEmotion",
            "HcpGambling",
            "HcpLanguage",
            "HcpMotor",
            "HcpRelational",
            "HcpSocial",
            "HcpWm",
        ]
    else:
        raise ValueError("Invalid value for tasktype")
    for dataset, tasks in dataset_task.items():
        dataset_root_path = os.path.join(data_root, dataset)
        for task in tasks:
            yield dataset, task, dataset_root_path


#### Extract Timeseries
# get the atlas
if os.path.exists(timeseries_path):
    print("Time series already extracted.")
    data = pd.read_pickle(timeseries_path)
else:
    atlas = datasets.fetch_atlas_schaefer_2018(
        data_dir=data_root, resolution_mm=1, n_rois=n_parcels
    )
    # use the atlas to extract time series for each task in parallel
    # get_time_series returns a dataframe with the time series for each task,
    # consisting of runs x subjects
    print("Time series extraction...")
    data = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(get_time_series)(
            task=task,
            atlas=atlas,
            data_root_path=dataset_root_path,
            dataset=dataset,
            trim_timeseries_to=trim_length,
        )
        for dataset, task, dataset_root_path in generator_dataset_task(
            dataset_task, tasktype
        )
    )
    # concatenate all the dataframes so we have a single dataframe with the
    # time series from all tasks
    data = pd.concat(data)
    data.reset_index(inplace=True, drop=True)
    # save the data
    data.to_pickle(timeseries_path)

#### Calculate Connectivity
# estimate the connectivity matrices for each cov estimator in parallel
# get_connectomes returns a dataframe with two columns each corresponding
# to the partial correlation and correlation connectome from each cov
# estimator
print("Connectivity estimation...")
data = Parallel(n_jobs=20, verbose=0)(
    delayed(get_connectomes)(cov, data, n_jobs) for cov in cov_estimators
)
# concatenate the dataframes so we have a single dataframe with the
# connectomes from all cov estimators
common_cols = ["time_series", "subject_ids", "run_labels", "tasks"]
data_ts = data[0][common_cols]
for df in data:
    df.drop(columns=common_cols, inplace=True)
data.append(data_ts)
data = pd.concat(data, axis=1)
data.reset_index(inplace=True, drop=True)
# save the data
data.to_pickle(fc_data_path)
