import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed, dump
import time
import os


def get_nan_indices(df):
    X = np.array(df["Graphical-Lasso partial correlation"].values.tolist())
    return np.where(np.isnan(X).all(axis=1))


def all_combinations(tasks, connectivity_measures):
    strings = []
    combinations = []
    for task_1 in tasks:
        for task_2 in tasks:
            possible_combinations = {
                f"{task_1}_{task_2}",
                f"{task_2}_{task_1}",
            }
            if possible_combinations & set(strings):
                already_done = True
            else:
                already_done = False
            if task_1 == task_2:
                continue
            elif task_1 != task_2 and not already_done:
                strings.append(f"{task_1}_{task_2}")
                combinations.append((task_1, task_2))

    for combination in combinations:
        for connectivity_measure in connectivity_measures:
            yield combination, connectivity_measure


def hcp_subject_fingerprinting_pairwisetasks(
    df, task_1, task_2, connectivity_measure
):
    all_scores = {}
    df_task1_task2 = df[df["tasks"].isin([task_1, task_2])]
    nan_indices = get_nan_indices(df_task1_task2)
    X = np.array(df_task1_task2[connectivity_measure].values.tolist())
    y = df_task1_task2["subject_ids"].to_numpy(dtype=object)
    groups = df_task1_task2["tasks"].to_numpy(dtype=object)
    X = np.delete(X, nan_indices, axis=0)
    y = np.delete(y, nan_indices, axis=0)
    groups = np.delete(groups, nan_indices, axis=0)

    n_groups = cv_splits = len(np.unique(groups))
    # set-up cross-validation scheme
    cv = GroupKFold(n_splits=n_groups, random_state=0, shuffle=True)
    classifier = LinearSVC(max_iter=100000, dual="auto")

    # cross-validate
    scores = cross_validate(
        classifier,
        X,
        y,
        groups=groups,
        cv=cv,
        n_jobs=1,
        return_train_score=True,
        return_estimator=True,
        scoring=["balanced_accuracy", "f1_macro"],
        return_indices=True,
        verbose=11,
    )
    return {
        "scores": scores,
        "mean_f1_macro": np.mean(scores["test_f1_macro"]),
        "mean_balanced_accuracy": np.mean(scores["test_balanced_accuracy"]),
        "task1": task_1,
        "task2": task_2,
        "connectivity_measure": connectivity_measure,
    }


# root = "/data/parietal/store3/work/haggarwa/connectivity/results/"
root = "/Users/himanshu/Desktop/ibc/connectivity/results/"
fc_data_path = os.path.join(
    root, "connectomes_nparcels-200_tasktype-domain_trim-None.pkl"
)

df = pd.read_pickle(fc_data_path)

df = df[df["dataset"] == "HCP900"]
df.reset_index(drop=True, inplace=True)

tasks = [
    "HcpEmotion",
    "HcpGambling",
    "HcpLanguage",
    "HcpMotor",
    "HcpRelational",
    "HcpSocial",
    "HcpWm",
]

connectivity_measures = [
    "Graphical-Lasso partial correlation",
    "Unregularized correlation",
    "Ledoit-Wolf correlation",
]

# all results
all_results = Parallel(n_jobs=10, verbose=11)(
    delayed(hcp_subject_fingerprinting_pairwisetasks)(
        df, task_1, task_2, connectivity_measure
    )
    for (task_1, task_2), connectivity_measure in all_combinations(
        tasks, connectivity_measures
    )
)
all_results = pd.DataFrame(all_results)

# Save the results
output_dir = f"hcp_subject_fingerprinting_pairwise_tasks_{time.strftime('%Y%m%d-%H%M%S')}"
output_path = os.path.join(root, output_dir)
os.makedirs(output_path, exist_ok=True)
# Save the results to a file
all_results.to_pickle(os.path.join(output_path, "all_results.pkl"))

# Print the results
print(all_results)
