"""This script renames all HCP and thelittleprince tasks to match corresponding IBC names"""

import pandas as pd
from glob import glob
import os

conversion = {
    "EMOTION": "HcpEmotion",
    "GAMBLING": "HcpGambling",
    "LANGUAGE": "HcpLanguage",
    "MOTOR": "HcpMotor",
    "RELATIONAL": "HcpRelational",
    "SOCIAL": "HcpSocial",
    "WM": "HcpWm",
    "lppFR": "LePetitPrince",
}


# Load the data
results = "/storage/store3/work/haggarwa/connectivity/results/"
# find all pickle files with connectomes
connectome_files = glob(os.path.join(results, "connectomes*.pkl"))

for connectome_file in connectome_files:
    print("\n", connectome_file)
    connectomes = pd.read_pickle(connectome_file)
    print("\nbefore")
    print(connectomes["tasks"].value_counts())

    connectomes["tasks"] = connectomes["tasks"].apply(
        lambda x: conversion.get(x, x) if x in conversion else x
    )
    print("\n\nafter")
    print(connectomes["tasks"].value_counts())

    connectomes.to_pickle(connectome_file)
    print("Saved to", connectome_file)
    print("-------------------------")
