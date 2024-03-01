import os
import pandas as pd
from PIL import Image
from datasets import Dataset
import datasets
from tqdm import tqdm

DATASET_PATH = f"{os.environ.get('DATASETS_PATH')}/Datasets_adults/FaceScape"
METADATA_PATH = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/orientation/metadata.csv"

def load_image_as_pil(file_path):
    return Image.open(file_path)

def get_prompt(orientation, expression):
    orientation = orientation[0].upper() + orientation[1:].lower() + " profile"
    expression = expression.split("_")[1]
    return f"{orientation} portrait of a {expression} asian adult" 

df = pd.read_csv(METADATA_PATH, index_col=0)


df_frontal = df[df["orientation"] == "frontal"]

images = []
condition_images = []
prompts = []
for idx, frontal_row in tqdm(df_frontal.iterrows()):
    frontal_img = load_image_as_pil(f"{DATASET_PATH}/{frontal_row['subject']}/{frontal_row['expression']}/{frontal_row['img']}.jpg")
    df_candidates = df[(df["subject"] == frontal_row["subject"]) & (df["orientation"] != "frontal")]
    for idx, lateral_row in df_candidates.iterrows():
        lateral_img = load_image_as_pil(f"{DATASET_PATH}/{lateral_row['subject']}/{lateral_row['expression']}/{lateral_row['img']}.jpg")
        images.append(lateral_img)
        condition_images.append(frontal_img)
        prompts.append(get_prompt(lateral_row['orientation'], lateral_row['expression']))

dataset_train = Dataset.from_dict({
    "image": images,
    "prompt": prompts,
    "conditioning_image": condition_images
})

dataset = datasets.DatasetDict({
        "train": dataset_train
})

dataset.save_to_disk(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/orientation/v1")