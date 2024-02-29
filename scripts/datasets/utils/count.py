import datasets
import os

dataset = datasets.load_from_disk(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/processed/alexg99-captioned_flickr_faces-custom_caption_v2")

print(len(list(filter(lambda x: "baby" in x, dataset['train']['custom_caption']))))