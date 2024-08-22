import os
import sys
import pandas as pd
import seaborn as sns
from nilearn import datasets
from joblib import Parallel, delayed
import numpy as np

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.fetching import get_ses_modality, get_confounds, get_niftis


def _update_data(data, motions, subject_ids, run_labels, tasks, dataset):
    """Update the data dictionary with the new time series, subject ids,
    run labels"""
    data["motion"].extend(motions)
    data["subject_ids"].extend(subject_ids)
    data["run_labels"].extend(run_labels)
    data["tasks"].extend(tasks)
    data["dataset"].extend(dataset)

    return data


#### INPUTS
# cache and root output directory
data_root = "/storage/store3/work/haggarwa/connectivity/data/"
# results directory
results = "/storage/store3/work/haggarwa/connectivity/results"
os.makedirs(results, exist_ok=True)
# number of jobs to run in parallel
n_jobs = 10
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
    # "HCP900": [
    #     "EMOTION",
    #     "GAMBLING",
    #     "LANGUAGE",
    #     "MOTOR",
    #     "RELATIONAL",
    #     "SOCIAL",
    #     "WM",
    # ],
    # "archi": [
    #     "ArchiStandard",
    #     "ArchiSpatial",
    #     "ArchiSocial",
    #     "ArchiEmotional",
    # ],
}


#### SETUP
# output data paths
output_path = os.path.join(
    results,
    f"motion_parameters.pkl",
)

data = {
    "motion": [],
    "subject_ids": [],
    "run_labels": [],
    "tasks": [],
    "dataset": [],
}

for dataset, tasks in dataset_task.items():
    data_root_path = os.path.join(data_root, dataset)
    print(f"Processing {dataset}")
    print(data_root_path)
    for task in tasks:
        subject_sessions, _ = get_ses_modality(
            task=task, data_root_path=data_root_path, dataset=dataset
        )
        motions = []
        subject_ids = []
        run_labels_ = []
        for subject, sessions in subject_sessions.items():
            for session in sorted(sessions):
                runs, run_labels = get_niftis(
                    task,
                    subject,
                    session,
                    data_root_path,
                    dataset,
                )
                for run, run_label in zip(runs, run_labels):
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
                        motion_diff = np.diff(confounds, axis=0)
                        # calculate framewise displacement
                        confounds = np.sum(
                            np.abs(motion_diff[:, 0:3])
                            + 50 * np.abs(motion_diff[:, 3:]),
                            axis=1,
                        )
                    motions.append(confounds)
                    subject_ids.append(subject)
                    run_labels_.append(run_label)
        tasks_ = [task for _ in range(len(motions))]
        datasets_ = [dataset for _ in range(len(motions))]

        data = _update_data(
            data, motions, subject_ids, run_labels_, tasks_, datasets_
        )


df = pd.DataFrame(data)
df.to_pickle(output_path)
