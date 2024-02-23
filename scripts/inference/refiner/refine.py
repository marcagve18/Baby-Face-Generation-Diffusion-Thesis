import argparse
from PIL import Image
import torch
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL
)

def load_image(path : str):
    with Image.open(path) as img:
        return img.convert("RGB")
    

def predict(args):
    image = load_image(args.image_path)
    negative_prompt = "teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant"

    seed = torch.random.initial_seed()

    print(image)
    generator = torch.manual_seed(seed)
    model_args = {
            "prompt": args.prompt,
            "image": image,
            "control_image": image,
            "strength": args.strength,
            "controlnet_conditioning_scale": 0.75,
            "negative_prompt": negative_prompt,
            "guidance_scale": 7,
            "generator": generator,
            "num_inference_steps": args.num_inference_steps,
            "guess_mode": False,
        }

    controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch.float16
        )
    
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema",
        torch_dtype=torch.float16
    )

    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            vae=vae,
            torch_dtype=torch.float16,
            controlnet=controlnet
        ).to("cuda")
    
    pipeline.safety_checker = None
    pipeline.enable_xformers_memory_efficient_attention()

    output = pipeline(**model_args).images[0]
    print(output)


def parse_args():
    parser = argparse.ArgumentParser(description='Refine the generated image')
    parser.add_argument(
        '--prompt', 
        type=str, 
        help='Prompt for image generation',
        required=True
    )
    parser.add_argument(
        '--image_path', 
        type=str, 
        default=None, 
        help='Image to be refined path',
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
        default=0.25, 
        help='Refinement strength'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    predict(args)
