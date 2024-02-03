from datasets import Dataset
import datasets
from PIL import Image
import os
import argparse


def load_images_from_folder(folder_path):
    file_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_paths.append(os.path.join(folder_path, filename))
    return file_paths

def load_image_as_pil(file_path):
    return Image.open(file_path)
    
def create_image_dataset(folder_path):
    file_paths = load_images_from_folder(folder_path)
    images = [load_image_as_pil(file_path) for file_path in file_paths]
    dataset = Dataset.from_dict({"image": images})
    return dataset

def run(images_path, output_name):
    image_dataset = create_image_dataset(images_path)
    dataset = datasets.DatasetDict({
        "train": image_dataset
    })

    dataset.save_to_disk(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/custom/only_images/{output_name}")
    
def parse_args():
    parser = argparse.ArgumentParser(description='Convert a folder with images to a HF dataset.')
    parser.add_argument(
        '--images_path', 
        type=str, 
        help='Path were the images are located',
        required=True
    )
    parser.add_argument(
        '--output_name', 
        type=str, 
        help='Output name of the dataset',
        required=True
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.images_path, args.output_name)