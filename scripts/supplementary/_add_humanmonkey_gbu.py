import pandas as pd
from glob import glob
import os

# Load the data
results = "/data/parietal/store3/work/haggarwa/connectivity/results/"
# find pickle file with connectomes for 200 parcels, naturalistic tasks,
# and no trimming
connectome_file = glob(
    os.path.join(results, "connectomes*200*natural*None.pkl")
)[0]
connectome = pd.read_pickle(connectome_file)

# load human monkey gbu connectomes
humanmonkey_gbu = pd.read_pickle(
    os.path.join(
        results,
        "before_review",
        "external_connectivity_20240125-104121",
        "connectomes_200_compcorr.pkl",
    )
)


humanmonkey_gbu["tasks"] = ["GoodBadUgly"] * len(humanmonkey_gbu)
humanmonkey_gbu["dataset"] = ["HumanMonkeyGBU"] * len(humanmonkey_gbu)

print(connectome.head())
print(connectome.columns)
print(humanmonkey_gbu.head())
print(humanmonkey_gbu.columns)

combined = pd.concat([connectome, humanmonkey_gbu])
combined.to_pickle(connectome_file)
