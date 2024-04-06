import torch
from PIL import Image
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerDiscreteScheduler,
    DDIMScheduler
)
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import cv2
import os
from datetime import datetime
import json
import numpy as np


def get_mask(img):

    img = np.array(img)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Create an empty image for contours
    img_contours = np.zeros(img.shape)
    # Draw the contours
    cv2.drawContours(img_contours, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convert the contours image to grayscale and then to binary
    img_contours_gray = cv2.cvtColor(img_contours.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, img_contours_binary = cv2.threshold(img_contours_gray, 128, 255, cv2.THRESH_BINARY)

    # Blur the binary image to get blurry edges. The larger the kernel, the blurrier the edges.
    kernel_size = (21, 21) # Using a large kernel for more blur
    img_contours_binary_blur = cv2.GaussianBlur(img_contours_binary, kernel_size, 0)

    img_contours_inverted = cv2.bitwise_not(img_contours_binary_blur)

    return Image.fromarray(img_contours_inverted)


realvis = "SG161222/Realistic_Vision_V5.1_noVAE"
my_model = "/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/models/sd_15_v5"

device = "cuda"
base_model = realvis
controlnet = "lllyasviel/control_v11f1p_sd15_depth"
ip_adapter_ckpt ="ip-adapter-faceid-plusv2_sd15.bin"
img_path = "/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/individuals/img_2024-03-07 12:55:52.994781_Frontal portrait of a smiling asian baby.png"
ip_adapter_path = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/ip-adapters/{ip_adapter_ckpt}"
vanilla = "runwayml/stable-diffusion-v1-5"


def load_image(path : str):
    with Image.open(path) as img:
        return img.convert("RGB")

scheduler = EulerDiscreteScheduler.from_pretrained(vanilla, subfolder="scheduler")

image = load_image(img_path)
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))
image_cv = cv2.imread(img_path)
faces = app.get(image_cv)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
face_image = face_align.norm_crop(image_cv, landmark=faces[0].kps, image_size=224) # you can also segment the face



noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

negative_prompt = "teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16
)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-ema",
    torch_dtype=torch.float16
)


seed = torch.random.initial_seed()

generator = torch.manual_seed(seed)

pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_model,
    vae=vae,
    torch_dtype=torch.float16,
    scheduler=scheduler,
    controlnet=controlnet,
    feature_extractor=None,
    safety_checker=None
).to("cuda")

pipeline.safety_checker = None
pipeline.enable_xformers_memory_efficient_attention()

ip_model = IPAdapterFaceIDPlus(pipeline, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", ip_adapter_path, device)

prompt = "Lateral portrait of a asian smiling baby, lateral view, profile, side"

rotated_img_3d = load_image("/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/tests/Screenshot 2024-03-09 at 13.07.41.png")
rotated_head_3d = load_image("/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/tests/Screenshot 2024-03-07 at 23.30.22.png")

model_args = {
    "prompt": prompt,
    "height": 512,
    "width": 512,
    "image": rotated_img_3d,
    "control_image": rotated_img_3d,
    "strength": 0.25,
    "controlnet_conditioning_scale": 1.0,
    "negative_prompt": negative_prompt,
    "guidance_scale": 5,
    "num_inference_steps": 100,
    "guess_mode": True,
    "num_samples": 1,
    "faceid_embeds": faceid_embeds, 
    "face_image": face_image,
    "s_scale": 1.0
}

image = ip_model.generate(**model_args)[0]
mask = get_mask(image)

# Save the figure
output_path = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/orientation"
current_datetime = datetime.now()
# Format the date and time as HH:MM:SS DD/MM/YY
filename = f"img_{current_datetime}_{prompt}"
image.save(f"{output_path}/{filename}.png")
mask.save(f"{output_path}/mask_{filename}.png")

with open(f"{output_path}/{filename}.json", 'w') as file:
    model_args["ip_adapter"] = ip_adapter_ckpt
    model_args["image"] = ""
    model_args["control_image"] = ""
    model_args["faceid_embeds"] = ""
    model_args["face_image"] = ""
    json.dump(model_args, file)