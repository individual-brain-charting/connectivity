import pandas as pd
from glob import glob
import os

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
    "thelittleprince": ["lppFR"],
}


def task_to_dataset(dataset_to_task):
    inverse = {}
    for k, v in dataset_to_task.items():
        for x in v:
            inverse.setdefault(x, k)
    return inverse


def fix_non_ibc_archi(run_label, task, dataset):
    if "Archi" in task:
        if "dir" in run_label:
            return "ibc"
        else:
            return "archi"
    else:
        return dataset


task_dataset = task_to_dataset(dataset_task)


# Load the data
results = "/storage/store3/work/haggarwa/connectivity/results/"
# find all pickle files with connectomes
connectome_files = glob(os.path.join(results, "connectomes*.pkl"))

for connectome_file in connectome_files:
    print("\n", connectome_file)
    connectomes = pd.read_pickle(connectome_file)
    print("\nbefore")
    print(connectomes["tasks"].value_counts())
    print(len(connectomes))

    connectomes["dataset"] = connectomes["tasks"].map(task_dataset)
    connectomes["dataset"] = connectomes.apply(
        lambda x: fix_non_ibc_archi(x["run_labels"], x["tasks"], x["dataset"]),
        axis=1,
    )
    print("\n\nafter")
    print(connectomes["dataset"].value_counts())
    print(connectomes["dataset"].value_counts().sum())

    connectomes.to_pickle(connectome_file)
    print("Saved to", connectome_file)
    print("-------------------------")
