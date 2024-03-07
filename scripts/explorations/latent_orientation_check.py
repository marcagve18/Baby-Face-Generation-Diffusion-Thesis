from diffusers.image_processor import VaeImageProcessor
import datasets
import os
from diffusers import AutoencoderKL
import torch
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

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
    with torch.no_grad():
        latents = vae.encode(input_img.to("cuda").to(torch.float16)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        return latents
    
p_colors = ['red', 'green', 'blue']  # Different colors for each type
def get_color(orientation):
    if orientation == "Left":
        return p_colors[0]
    elif orientation == "Right":
        return p_colors[1]
    else: 
        return p_colors[2]

print("Loading VAE")
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to("cuda", dtype=torch.float16)
print("VAE loaded")

added_imgs = []
encoded_imgs = []
colors = []
print("Encoding")
subset = dataset['train'].select(range(350))
print("subset: ", subset)
for row in tqdm(subset):
    if row['image'] not in added_imgs:
        colors.append(get_color(row['prompt'].split(" ")[0]))
        added_imgs.append(row['image'])
        encoded_imgs.append(encode_img(row['image']))

    if row['conditioning_image'] not in added_imgs:   
        colors.append(get_color("Frontal")) 
        added_imgs.append(row['conditioning_image'])
        encoded_imgs.append(encode_img(row['conditioning_image']))
print("Encoded")


# Concatenate all tensors in the list into a single tensor of shape [10, 4, 64, 64]
all_tensors = torch.cat(encoded_imgs, dim=0).to("cpu")

# Reshape concatenated tensor to [10, 4*64*64] for PCA
# This treats each channel of each image as a separate feature, with each image flattened
all_tensors_reshaped = all_tensors.view(all_tensors.shape[0], -1).numpy()

# Initialize and apply PCA
pca = PCA(n_components=2)  # Adjust n_components based on your needs
pca_result = pca.fit_transform(all_tensors_reshaped)

# Plotting the PCA result
labels = [prompt.split(" ")[0] for prompt in subset['prompt']]
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, marker='o', edgecolor='black', s=50)


legend_elements = [Line2D([0], [0], marker='o', color=p_colors[0], label='Left',
                            markerfacecolor='red', markersize=10),
                    Line2D([0], [0], marker='o', color=p_colors[1], label='Right',
                            markerfacecolor='green', markersize=10),
                    Line2D([0], [0], marker='o', color=p_colors[2], label='Frontal',
                            markerfacecolor='green', markersize=10)]
plt.legend(handles=legend_elements, loc='best')

    
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

plt_path = "/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/plots/pca/pca_result_plot.png"
plt.savefig(plt_path)