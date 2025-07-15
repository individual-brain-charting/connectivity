import sys
import os
from tqdm import tqdm
from nilearn.image import load_img
import pandas as pd
from joblib import Parallel, delayed
from glob import glob

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.fetching import get_ses_modality, get_niftis

data_root = "/data/parietal/store3/work/haggarwa/connectivity/data/"

dataset_task = {
    "ibc": [
        "GoodBadUgly",
        "MonkeyKingdom",
        "Raiders",
        "RestingState",
        "Mario",
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
        "LePetitPrince",
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


def generator_dataset_task(dataset_task):
    for dataset, tasks in dataset_task.items():
        dataset_root_path = os.path.join(data_root, dataset)
        for task in tasks:
            yield dataset, task, dataset_root_path


def summarise(
    dataset, task, dataset_root_path, data_root=data_root, verbose=False
):
    infos = []
    sub_ses, _ = get_ses_modality(
        task=task, data_root_path=dataset_root_path, dataset=dataset
    )
    if verbose:
        print("sub_ses", sub_ses)
    for subject, sessions in tqdm(
        sub_ses.items(), desc=task, total=len(sub_ses)
    ):
        if verbose:
            print("subject, sessions", subject, sessions)
        for session in sorted(sessions):
            runs, run_labels = get_niftis(
                task=task,
                subject=subject,
                session=session,
                data_root_path=dataset_root_path,
                dataset=dataset,
            )
            if verbose:
                print("runs, run_labels", runs, run_labels)
            for run, run_label in zip(runs, run_labels):
                try:
                    img = load_img(run)
                    print(task, subject, run_label, img.shape[-1])
                    info = {
                        "dataset": dataset,
                        "task": task,
                        "subject": subject,
                        "session": session,
                        "run": run_label,
                        "n_trs": img.shape[-1],
                        "dim": img.shape,
                        "path": run,
                        "voxel_sizes": img.header.get_zooms(),
                    }
                except EOFError as e:
                    print(e)
                    info = {
                        "dataset": dataset,
                        "task": task,
                        "subject": subject,
                        "session": session,
                        "run": run_label,
                        "n_trs": None,
                        "dim": None,
                        "path": run,
                        "voxel_sizes": None,
                    }
                    continue
                infos.append(info)
    df = pd.DataFrame(infos)
    summary_file = f"{dataset}_{task}_summary.csv"
    df.to_csv(os.path.join(data_root, summary_file), index=False)

    return df


dfs = Parallel(n_jobs=50, verbose=2)(
    delayed(summarise)(*args) for args in generator_dataset_task(dataset_task)
)

# for dataset, task, dataset_root in generator_dataset_task(dataset_task):
#     summarise(dataset, task, dataset_root, verbose=True)

# load the full summary
df = pd.read_csv(os.path.join(data_root, "full_summary.csv"))

# drop last 2 runs of Raiders, GoodBadUgly and only for sub-15 in RestingState
df = df.drop(df[(df["task"] == "Raiders") & (df["run"] == "run-10")].index)
df = df.drop(df[(df["task"] == "Raiders") & (df["run"] == "run-09")].index)
df = df.drop(df[(df["task"] == "GoodBadUgly") & (df["run"] == "run-18")].index)
# also drop run-01, run-02 of GoodBadUgly
df = df.drop(df[(df["task"] == "GoodBadUgly") & (df["run"] == "run-01")].index)
df = df.drop(df[(df["task"] == "GoodBadUgly") & (df["run"] == "run-02")].index)
df = df.drop(
    df[
        (df["task"] == "RestingState")
        & (df["run"] == "dir-pa")
        & (df["subject"] == "sub-15")
    ].index
)

# filter only the IBC naturalistic tasks
df_ibc_natural = df[
    df["task"].isin(
        [
            "GoodBadUgly",
            "MonkeyKingdom",
            "Raiders",
            "RestingState",
            "LePetitPrince",
            "Mario",
        ]
    )
]
# get the minimum number of TRs by task
print(df.groupby(["dataset", "task"])["n_trs"].min())
print(df_ibc_natural.groupby(["task"])["n_trs"].min())
