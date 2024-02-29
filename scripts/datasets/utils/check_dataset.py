import datasets
import os

dataset = datasets.load_from_disk(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/custom/only_images/Baby_Face_Clean")

print(dataset)