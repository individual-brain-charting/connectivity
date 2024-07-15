"""Utility functions for functional connectivity estimation"""

import numpy as np
import pandas as pd
from nilearn.connectome import sym_matrix_to_vec
from nilearn.image import high_variance_confounds
from nilearn.maskers import NiftiLabelsMasker
from sklearn.base import clone
from sklearn.covariance import (
    GraphicalLassoCV,
    GraphicalLasso,
    LedoitWolf,
    EmpiricalCovariance,
    empirical_covariance,
    shrunk_covariance,
)
from tqdm import tqdm
import itertools
from .fetching import get_ses_modality, get_niftis, get_confounds, get_tr


def _update_data(data, all_time_series, subject_ids, run_labels, tasks):
    """Update the data dictionary with the new time series, subject ids,
    run labels"""
    data["time_series"].extend(all_time_series)
    data["subject_ids"].extend(subject_ids)
    data["run_labels"].extend(run_labels)
    data["tasks"].extend(tasks)

    return data


def _trim_timeseries(timeseries, n):
    """Trim the time series to the given length"""
    return timeseries[:n, :]


def get_time_series(task, atlas, data_root_path, dataset="ibc"):
    """Use NiftiLabelsMasker to extract time series from nifti files.

    Parameters
    ----------
    tasks : list
        List of tasks to extract time series from.
    atlas : atlas object
        Atlas to use for extracting time series.
    data_root_path : str
        Path to data root directory.
    dataset : str, optional
        Name of the dataset, by default "ibc". Could also be "HCP9000" or
        "archi" or "thelittleprince".

    Returns
    -------
    pandas DataFrame
        DataFrame containing the time series, subject ids, run labels,
        and tasks.
    """
    data = {
        "time_series": [],
        "subject_ids": [],
        "run_labels": [],
        "tasks": [],
    }
    repetition_time = get_tr(task, dataset)
    print(f"Getting time series for {task}...")
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize="zscore_sample",
        low_pass=0.2,
        high_pass=0.01,
        t_r=repetition_time,
        verbose=0,
        # memory=Memory(location=cache),
        memory_level=0,
        n_jobs=1,
    ).fit()
    subject_sessions, _ = get_ses_modality(
        task=task, data_root_path=data_root_path, dataset=dataset
    )
    all_time_series = []
    subject_ids = []
    run_labels_ = []
    for subject, sessions in tqdm(
        subject_sessions.items(), desc=task, total=len(subject_sessions)
    ):
        for session in sorted(sessions):
            runs, run_labels = get_niftis(
                task,
                subject,
                session,
                data_root_path,
                dataset,
            )
            for run, run_label in zip(runs, run_labels):
                print(task)
                print(subject, session)
                print(run_label)
                if dataset == "thelittleprince":
                    confounds = None
                else:
                    confounds = get_confounds(
                        task,
                        run_label,
                        subject,
                        session,
                        data_root_path,
                        dataset,
                    )
                    confounds = np.loadtxt(confounds)
                    # some HCP subjects throw EOFError
                    try:
                        compcor_confounds = high_variance_confounds(run)
                    except EOFError as e:
                        print(e, run)
                        with open("log_EOFError.txt", "w") as f:
                            f.write(f"{run}\n")
                        continue
                    # archi, ArchiSpatial, sub-01 confounds had an extra row
                    try:
                        confounds = np.hstack((confounds, compcor_confounds))
                    except ValueError as e:
                        with open("log_ValueError.txt", "w") as f:
                            f.write(f"{e}\n")
                            f.write(f"{run}\n")
                time_series = masker.transform(run, confounds=confounds)
                all_time_series.append(time_series)
                subject_ids.append(subject)
                run_labels_.append(run_label)

    tasks_ = [task for _ in range(len(all_time_series))]

    data = _update_data(
        data, all_time_series, subject_ids, run_labels_, tasks_
    )
    return pd.DataFrame(data)


def calculate_connectivity(
    X, cov_estimator, data, ind, trim_timeseries_to=None
):
    """Fit given covariance estimator to data and return correlation
     and partial correlation.

    Parameters
    ----------
    X : numpy array
        Time series data.
    cov_estimator : sklearn estimator
        Covariance estimator to fit to data.
    trim_timeseries_to : int, optional
        Length to trim time series to, by default None
    Returns
    -------
    tuple of numpy arrays
        First array is the correlation matrix, second array is the partial
    """
    # get the connectivity measure
    cov_estimator_ = clone(cov_estimator)

    # fix error ValueError: Buffer dtype mismatch, expected 'const double' but
    # got 'float'
    X = X.astype(np.double)

    # trim the time series
    if trim_timeseries_to is not None:
        X = _trim_timeseries(X, trim_timeseries_to)

    # GraphicalLassoCV throws Non SPD results error, needs longer time series
    try:
        cv = cov_estimator_.fit(X)
        cv_correlation = sym_matrix_to_vec(
            cv.covariance_, discard_diagonal=True
        )
        cv_partial = sym_matrix_to_vec(-cv.precision_, discard_diagonal=True)
    except FloatingPointError as e:
        print(e)
        cv_correlation = np.nan
        cv_partial = np.nan
        with open("log_NonSPD.txt", "w") as f:
            f.write(f"{data.loc["subject_ids", ind]}\t")
            f.write(f"{data.loc["run_labels", ind]}\t")
            f.write(f"{data.loc["tasks", ind]}\n")

    return (cv_correlation, cv_partial)


def get_connectomes(cov, data, n_jobs, trim_timeseries_to=None):
    """Wrapper function to calculate connectomes using different covariance
    estimators. Selects appropriate covariance estimator based on the
    given string and adds the connectomes to the given data dataframe."""
    # covariance estimator
    if cov == "Graphical-Lasso":
        cov_estimator = GraphicalLassoCV(n_jobs=n_jobs, assume_centered=True)
    elif cov == "Ledoit-Wolf":
        cov_estimator = LedoitWolf(assume_centered=True)
    elif cov == "Unregularized":
        cov_estimator = EmpiricalCovariance(assume_centered=True)
    time_series = data["time_series"].tolist()
    connectomes = []
    count = 0
    for ts in tqdm(time_series, desc=cov, leave=True):
        connectome = calculate_connectivity(
            ts, cov_estimator, data, count, trim_timeseries_to
        )
        connectomes.append(connectome)
        count += 1
    correlation = np.asarray([connectome[0] for connectome in connectomes])
    partial_correlation = np.asarray(
        [connectome[1] for connectome in connectomes]
    )
    data[f"{cov} correlation"] = correlation.tolist()
    data[f"{cov} partial correlation"] = partial_correlation.tolist()

    return data
