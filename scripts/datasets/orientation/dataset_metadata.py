import os
import pandas as pd
import json

CAMERAS_ORIENTATIONS = {
    "frontal" : [
        "344054000909", "351035021127", "351035012642"
    ],
    "left" : [
        "344054001456", "344054001450", "351035018540"
    ],
    "right" : [
        "344054001454", "361035004896", "361035001964"
    ]
}

DATASET_PATH = f"{os.environ.get('DATASETS_PATH')}/Datasets_adults/FaceScape"

def load_parameters(path):
    with open(f"{path}/params.json", 'r') as f:
        params = json.load(f) # read parameters
        return params

expression = "1_neutral"

folders = os.listdir(DATASET_PATH)

df = pd.DataFrame(columns=["subject", "expression", "orientation", "img", "sn"])

for subject in folders:
    if expression in os.listdir(f"{DATASET_PATH}/{subject}") and "params.json" in os.listdir(f"{DATASET_PATH}/{subject}/{expression}"):
        subject_params = load_parameters(f"{DATASET_PATH}/{subject}/{expression}")
        subject_cameras_sn = {k: v for k, v in subject_params.items() if "_sn" in k}
        
        for key, sn in subject_cameras_sn.items():
            img_number = key.split("_")[0]
            if sn in CAMERAS_ORIENTATIONS["frontal"]:
                df.loc[len(df.index)] = [subject, expression, "frontal", img_number, sn]
            elif sn in CAMERAS_ORIENTATIONS["left"]:
                df.loc[len(df.index)] = [subject, expression, "left", img_number, sn]
            elif sn in CAMERAS_ORIENTATIONS["right"]:
                df.loc[len(df.index)] = [subject, expression, "right", img_number, sn]

df.to_csv(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/orientation/metadata.csv")
