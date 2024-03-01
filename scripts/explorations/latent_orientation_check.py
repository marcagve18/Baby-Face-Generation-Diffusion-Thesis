from diffusers.image_processor import VaeImageProcessor
import datasets
import os
from diffusers import AutoencoderKL
import torch
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA



dataset = datasets.load_from_disk(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/orientation/v1")

train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def encode_img(input_img):
    input_img = VaeImageProcessor().preprocess(image=input_img, height=512, width=512)
    latents = vae.encode(input_img).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

added_imgs = []
encoded_imgs = []
print("Encoding")
for row in dataset['train'].select(range(100)):
    if row['image'] not in added_imgs:
        added_imgs.append(row['image'])
        encoded_imgs.append(encode_img(row['image']))

    if row['conditioning_image'] not in added_imgs:    
        added_imgs.append(row['conditioning_image'])
        encoded_imgs.append(encode_img(row['conditioning_image']))
print("Encoded")


# Concatenate all tensors in the list into a single tensor of shape [10, 4, 64, 64]
all_tensors = torch.cat(encoded_imgs, dim=0)

# Reshape concatenated tensor to [10, 4*64*64] for PCA
# This treats each channel of each image as a separate feature, with each image flattened
all_tensors_reshaped = all_tensors.view(all_tensors.shape[0], -1).numpy()

# Initialize and apply PCA
pca = PCA(n_components=2)  # Adjust n_components based on your needs
pca_result = pca.fit_transform(all_tensors_reshaped)

print(pca_result)