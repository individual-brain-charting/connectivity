"""Pipeline to estimate functional connectivity matrices"""

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
n_parcels = 400  # or 200
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

# connectivity data path
if n_parcels == 400:
    # with compcorr
    fc_data_path = os.path.join(cache, "connectomes_400_comprcorr")
elif n_parcels == 200:
    # with compcorr
    fc_data_path = os.path.join(cache, "connectomes_200_comprcorr")

#### CALCULATE CONNECTIVITY
# get the atlas
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
# use the atlas to extract time series for each task in parallel
# get_time_series returns a dataframe with the time series for each task,
# consisting of runs x subjects
print("Time series extraction...")
data = Parallel(n_jobs=n_jobs, verbose=0)(
    delayed(get_time_series)(task, atlas, cache) for task in tasks
)
# concatenate all the dataframes so we have a single dataframe with the
# time series from all tasks
data = pd.concat(data)
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
