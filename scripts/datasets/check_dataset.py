import datasets
import os

dataset = datasets.load_from_disk(f"{os.environ.get("HOME_PATH")}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/raw/alexg99-captioned_flickr_faces-custom_caption")

print(dataset['custom_caption'][95])