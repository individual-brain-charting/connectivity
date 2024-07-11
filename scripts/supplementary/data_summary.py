import sys
import os
from tqdm import tqdm
from nilearn.image import load_img
import pandas as pd
from joblib import Parallel, delayed

# add utils to path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.fetching import get_ses_modality, get_niftis

data_root = "/storage/store3/work/haggarwa/connectivity/data/"

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
