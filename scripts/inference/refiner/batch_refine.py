import argparse
from PIL import Image
import torch
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL
)
from datetime import datetime
import os

def filter_images(path : str):
    refined_imgs = os.listdir(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/refiner-comparison/refined")
    all_imgs = os.listdir(path)
    candidate_imgs = list(filter(lambda filename: not filename.startswith("metadata") and f"refined_{filename}" not in refined_imgs and filename.endswith(".png"), all_imgs))
    return candidate_imgs

def load_images(path : str):
    with Image.open(path) as img:
        return img.convert("RGB")
    

def predict(args):
    candidate_imgs = filter_images(args.images_path)

    images = list(map(
        lambda filename: 
         [ 
             load_images(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/refiner-comparison/no-refiner/{filename}"), 
            filename.split("_")[-1].split(".")[0],
            filename
          ],
        candidate_imgs))

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

    base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model,
        vae=vae,
        torch_dtype=torch.float16,
        controlnet=controlnet
    ).to("cuda")
    
    pipeline.safety_checker = None
    pipeline.enable_xformers_memory_efficient_attention()


    for img, prompt, filename in images:
        model_args = {
            "prompt": prompt,
            "image": img,
            "control_image": img,
            "strength": args.strength,
            "controlnet_conditioning_scale": 0.9,
            "negative_prompt": negative_prompt,
            "guidance_scale": 7,
            "generator": generator,
            "num_inference_steps": args.num_inference_steps,
            "guess_mode": False,
        }
        output = pipeline(**model_args).images[0]
        output.save(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/refiner-comparison/refined/refined_{filename}")


def parse_args():
    parser = argparse.ArgumentParser(description='Refine the generated image')
    parser.add_argument(
        '--images_path', 
        type=str, 
        default=None, 
        help='Path where images are located',
        required=True
    )
    parser.add_argument(
        '--num_inference_steps', 
        type=int, 
        default=20, 
        help='Number of inference steps'
    )
    parser.add_argument(
        '--strength', 
        type=int, 
        default=0.12, 
        help='Refinement strength'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    predict(args)
