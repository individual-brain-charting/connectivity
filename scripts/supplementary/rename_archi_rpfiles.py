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

data_root = "/storage/store3/work/haggarwa/connectivity/data/"

protocol = {
    "emotional": "ArchiEmotional",
    "localizer": "ArchiStandard",
    "parietal": "ArchiSpatial",
    "social": "ArchiSocial",
}
rp_files = glob(os.path.join(data_root, "archi", "rpfiles", "rp*.txt"))

for f in rp_files:
    # rename the file
    split_name = os.path.basename(f).split("_")
    subject_number = int(split_name[3])
    task = protocol[split_name[5]]
    new_name = f"rp_sub-{subject_number:02}_task-{task}.txt"

    os.rename(f, os.path.join(data_root, "archi", "rpfiles", new_name))
    print(f, new_name)
