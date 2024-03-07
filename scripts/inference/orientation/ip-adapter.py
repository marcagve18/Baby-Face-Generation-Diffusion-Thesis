import torch
from PIL import Image
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerDiscreteScheduler
)
from insightface.app import FaceAnalysis
import cv2
import os

device = "cuda"
base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
controlnet = "lllyasviel/control_v11f1p_sd15_depth"
ip_adapter_ckpt ="ip-adapter-faceid_sd15.bin"
img_path = "/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/individuals/img_2024-03-07 12:55:52.994781_Frontal portrait of a smiling asian baby.png"
ip_adapter_path = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/ip-adapters/{ip_adapter_ckpt}"
vanilla = "runwayml/stable-diffusion-v1-5"


def load_image(path : str):
    with Image.open(path) as img:
        return img.convert("RGB")

scheduler = EulerDiscreteScheduler.from_pretrained(vanilla, subfolder="scheduler")

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))
image_cv = cv2.imread(img_path)
faces = app.get(image_cv)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)


negative_prompt = "teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant"

seed = torch.random.initial_seed()

generator = torch.manual_seed(seed)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile",
    torch_dtype=torch.float16
)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-ema",
    torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    vae=vae,
    scheduler=scheduler,
    feature_extractor=None,
    safety_checker=None
).to("cuda")


ip_model = IPAdapterFaceID(pipeline, ip_adapter_path, device)

prompt = "Lateral portrait of a asian smiling baby, lateral view, profile, side"

images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=1, width=512, height=512, num_inference_steps=20
)

images[0].save("/home/maguilar/TFG/out.png")
