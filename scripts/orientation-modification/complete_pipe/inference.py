import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

rembg_session = rembg.new_session()

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

device="cpu"
model_name = "stabilityai/TripoSR"

model = TSR.from_pretrained(
    model_name,
    config_name="config.yaml",
    weight_name="model.ckpt")
model.renderer.set_chunk_size(0)
model.to(device)


def get_3d_object(model, input_image, foreground_ratio = 0.75):
    # PREPROCESSING #
    image = input_image.convert("RGB")
    image = remove_background(image, rembg_session)
    image = resize_foreground(image, foreground_ratio)
    
    # Fill white background
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))

    print("preprocessing finished, generating letsgo")
    # GENERATION # 
    scene_codes = model(image, device=device)
    print("out1")
    mesh = model.extract_mesh(scene_codes, resolution=320)[0]
    print("out2")
    mesh = to_gradio_3d_orientation(mesh)
    mesh.apply_scale([-1, 1, 1])


    return mesh


def predict(args):
    image = Image.open(args.path)

    mesh = get_3d_object(model, image)
    mesh.export(f"mesh.ply")

    print(mesh)

def parse_args():
    parser = argparse.ArgumentParser(description='Rotate a given face')
    parser.add_argument(
        '--path', 
        type=str, 
        help='Path of the image to be rotated',
        required=True
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    predict(args)