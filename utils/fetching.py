"""Utility functions for fetching data"""

import os
from glob import glob
import pandas as pd
from ibc_public.utils_data import get_subject_session


def get_tr(task, dataset="ibc"):
    """Get repetition time for the given task and dataset

    Parameters
    ----------
    task : str
        Name of the task. Could be "RestingState", "GoodBadUgly", "Raiders",
        "MonkeyKingdom", "Mario" if dataset is "ibc". If dataset is "HCP900",
         all tasks have a repetition time of 0.72.
    dataset : str, optional
        Which dataset to use, by default "ibc", could also be "HCP900"

    Returns
    -------
    float or int
        Repetition time for the given task

    Raises
    ------
    ValueError
        If the task is not recognized
    """
    if dataset == "ibc":
        if task == "RestingState":
            repetition_time = 0.76
        elif task in [
            "GoodBadUgly",
            "Raiders",
            "MonkeyKingdom",
            "Mario",
            "LePetitPrince",
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
        ]:
            repetition_time = 2
        else:
            raise ValueError(f"Dont know the TR for task {task}")
    elif dataset == "HCP900":
        repetition_time = 0.72
    elif dataset == "archi":
        repetition_time = 2.4
    elif dataset == "thelittleprince":
        repetition_time = 2

    return repetition_time


def get_niftis(task, subject, session, data_root_path, dataset="ibc"):
    """Get nifti files of preprocessed BOLD data for the given task, subject,
    session and dataset.

    Parameters
    ----------
    task : str
        Name of the task
    subject : str
        subject id
    session : str
        session number
    dataset : str, optional
        which dataset to use, by default "ibc", could also be "HCP900"

    Returns
    -------
    list, list
        List of paths to nifti files, list of run labels for each nifti file
        eg. "run-01", "run-02", etc.
    """
    if dataset == "ibc":
        _run_files = glob(
            os.path.join(
                data_root_path,
                "derivatives",
                subject,
                session,
                "func",
                f"wrdc*{task}*.nii.gz",
            )
        )
        run_labels = []
        run_files = []
        for run in _run_files:
            run_label = os.path.basename(run).split("_")[-2]
            run_num = run_label.split("-")[-1]
            # skip repeats of run-01, run-02, run-03 done at the end of
            # the sessions in Raiders and GoodBadUgly
            if task == "Raiders" and int(run_num) > 8:
                continue
            # also skip short runs at the beginning and end of GoodBadUgly
            elif task == "GoodBadUgly" and (
                int(run_num) > 17 or int(run_num) < 3
            ):
                continue

            run_labels.append(run_label)
            run_files.append(run)
    elif dataset == "HCP900":
        run_files = glob(
            os.path.join(
                data_root_path,
                subject,
                "MNINonLinear",
                "Results",
                session,
                f"{session}.nii.gz",
            )
        )
        run_labels = []
        for run in run_files:
            direction = session.split("_")[2]
            if task == "REST":
                rest_ses = session.split("_")[1]
                if direction == "LR" and rest_ses == "REST1":
                    run_label = "run-01"
                elif direction == "RL" and rest_ses == "REST1":
                    run_label = "run-02"
                elif direction == "LR" and rest_ses == "REST2":
                    run_label = "run-03"
                elif direction == "RL" and rest_ses == "REST2":
                    run_label = "run-04"
            else:
                if direction == "LR":
                    run_label = "run-01"
                elif direction == "RL":
                    run_label = "run-02"
            run_labels.append(run_label)
    elif dataset == "archi":
        run_files = glob(
            os.path.join(
                data_root_path,
                "derivatives",
                subject,
                "func",
                f"*{task}*.nii.gz",
            )
        )
        run_labels = ["run-01" for _ in run_files]
    elif dataset == "thelittleprince":
        run_files = glob(
            os.path.join(
                data_root_path,
                "ds003643",
                "derivatives",
                subject,
                "func",
                "*.nii.gz",
            )
        )
        run_labels = []
        for run in run_files:
            run_label = os.path.basename(run).split("_")[2]
            run_labels.append(run_label)
    else:
        raise ValueError(f"Cant find niftis for dataset {dataset}")
    return run_files, run_labels


def get_confounds(
    task, run_label, subject, session, data_root_path, dataset="ibc"
):
    """Get confounds file for the given task, run number, subject, session
    and dataset.

    Parameters
    ----------
    task : str
        Name of the task
    run_label : str
        Run label of the nifti file
    subject : str
        subject id
    session : str
        session number
    dataset : str, optional
        name of the dataset, by default "ibc". Could also be "HCP900" or
        "archi" or "thelittleprince"

    Returns
    -------
    str
        Path to the confounds file
    """
    if dataset == "ibc":
        return glob(
            os.path.join(
                data_root_path,
                "derivatives",
                subject,
                session,
                "func",
                f"rp*{task}*{run_label}_bold*",
            )
        )[0]
    elif dataset == "HCP900":
        return glob(
            os.path.join(
                data_root_path,
                subject,
                "MNINonLinear",
                "Results",
                session,
                "Movement_Regressors_dt.txt",
            )
        )[0]
    elif dataset == "archi":
        return glob(
            os.path.join(
                data_root_path,
                "rpfiles",
                f"rp_{subject}*{task}*.txt",
            )
        )[0]
    elif dataset == "thelittleprince":
        raise ValueError("No confounds available for thelittleprince")
    else:
        raise ValueError(f"Cant find confounds for dataset {dataset}")


def _find_hcp_subjects(session_names, data_root_path):
    """Find HCP subjects with the given session names"""
    # load csv file with subject ids and task availability
    df = pd.read_csv(os.path.join(data_root_path, "unrestricted_hcp_s900.csv"))
    df = df[df["3T_Full_MR_Compl"] == True]
    subs = list(df["Subject"].astype(str))
    sub_ses = {}
    for sub in subs:
        sub_ses[sub] = session_names

    return sub_ses


def get_ses_modality(task, data_root_path, dataset="ibc"):
    """Get session numbers and modality for given task

    Parameters
    ----------
    task : str
        name of the task
    dataset : str
        name of the dataset, can be ibc or hcp

    Returns
    -------
    sub_ses : dict
        dictionary with subject as key and session number as value
    modality : str
        modality of the task
    """
    if dataset == "ibc":
        if task == "GoodBadUgly":
            # session names with movie task data
            session_names = ["BBT1", "BBT2", "BBT3"]
        elif task == "MonkeyKingdom":
            # session names with movie task data
            session_names = ["monkey_kingdom"]
        elif task == "Raiders":
            # session names with movie task data
            session_names = ["raiders1", "raiders2"]
        elif task == "RestingState":
            # session names with RestingState state task data
            session_names = ["mtt1", "mtt2"]
        elif task == "DWI":
            # session names with diffusion data
            session_names = ["anat1"]
        elif task == "Mario":
            # session names with mario gameplay data
            session_names = ["mario1"]
        elif task in [
            "ArchiStandard",
            "ArchiSpatial",
            "ArchiSocial",
            "ArchiEmotional",
        ]:
            session_names = ["archi"]
        elif task in ["HcpEmotion", "HcpGambling", "HcpLanguage", "HcpMotor"]:
            session_names = ["hcp1"]
        elif task in ["HcpRelational", "HcpSocial", "HcpWm"]:
            session_names = ["hcp2"]
        elif task == "LePetitPrince":
            session_names = ["lpp1", "lpp2"]
        else:
            raise ValueError(f"Unknown ibc task {task}")
        # get session numbers for each subject
        # returns a list of tuples with subject and session number
        subject_sessions = sorted(get_subject_session(session_names))
        # convert the tuples to a dictionary with subject as key and session
        # number as value
        sub_ses = {}
        # for dwi, with anat1 as session_name, get_subject_session returns
        # wrong session number for sub-01 and sub-15
        # setting it to ses-12 for these subjects
        if task == "DWI":
            modality = "structural"
            sub_ses = {
                subject_session[0]: (
                    "ses-12"
                    if subject_session[0] in ["sub-01", "sub-15"]
                    else subject_session[1]
                )
                for subject_session in subject_sessions
            }
        else:
            # for fMRI tasks, for one of the movies, ses no. 13 pops up for
            # sub-11 and sub-12, so skipping that
            modality = "functional"
            for subject_session in subject_sessions:
                if (
                    subject_session[0] in ["sub-11", "sub-12"]
                    and subject_session[1] == "ses-13"
                ):
                    continue
                # initialize a subject as key and an empty list as the value
                # and populate the list with session numbers
                # try-except block is used to avoid overwriting the list
                # for subject
                try:
                    sub_ses[subject_session[0]]
                except KeyError:
                    sub_ses[subject_session[0]] = []
                sub_ses[subject_session[0]].append(subject_session[1])

    elif dataset == "HCP900":
        hcp_tasks = [
            "EMOTION",
            "GAMBLING",
            "LANGUAGE",
            "MOTOR",
            "RELATIONAL",
            "SOCIAL",
            "WM",
        ]
        if task == "REST":
            # session names with RestingState state task data
            session_names = ["rfMRI_REST1_LR", "rfMRI_REST2_RL"]
        elif task in hcp_tasks:
            # session names with HCP task data
            session_names = [f"tfMRI_{task}_LR", f"tfMRI_{task}_RL"]
        else:
            raise ValueError(f"Unknown HCP task {task}")
        modality = "functional"
        # create dictionary with subject as key and session number as value
        sub_ses = _find_hcp_subjects(session_names, data_root_path)
    elif dataset == "archi":
        archi_tasks = [
            "ArchiStandard",
            "ArchiSpatial",
            "ArchiSocial",
            "ArchiEmotional",
        ]
        if task in archi_tasks:
            # no sessions for archi
            # just need to get all subject ids
            subs = glob(os.path.join(data_root_path, "derivatives", "sub-*"))
            subs = [os.path.basename(sub) for sub in subs]
            sub_ses = {}
            for sub in subs:
                sub_ses[sub] = ["ses-01"]
        else:
            raise ValueError(f"Unknown archi task {task}")
        modality = "functional"
    elif dataset == "thelittleprince":
        if task == "lppFR":
            # no sessions for thelittleprince
            # but only need the french subject ids
            subs = glob(
                os.path.join(
                    data_root_path, "ds003643", "derivatives", "sub-FR*"
                )
            )
            subs = [os.path.basename(sub) for sub in subs]
            sub_ses = {}
            for sub in subs:
                sub_ses[sub] = ["ses-01"]
        else:
            raise ValueError(f"Unknown thelittleprince task {task}")
        modality = "functional"
    else:
        raise ValueError(f"Cant find session numbers for dataset {dataset}")
    return sub_ses, modality
