import datasets
import os

dataset_t = datasets.load_from_disk(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/custom/captioned/Baby_Face_Clean")

dataset = datasets.DatasetDict({
    "train": dataset_t
})


dataset.push_to_hub("marcagve18/Baby_Face_Clean", private=True)