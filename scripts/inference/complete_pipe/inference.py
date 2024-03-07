import argparse
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL
)
from datetime import datetime
import os
import matplotlib.pyplot as plt
    

def predict(args):
    seed = torch.random.initial_seed()
    generator = torch.manual_seed(seed)
    vanilla = "runwayml/stable-diffusion-v1-5"

    print("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/{args.model}", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    print("Model loaded")
    scheduler = EulerDiscreteScheduler.from_pretrained(vanilla, subfolder="scheduler")


    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    pipeline.safety_checker = None


    image_finetuned = pipeline(
        prompt=args.prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=5,
        scheduler=scheduler, 
        generator=generator).images[0]

    # Create a Matplotlib figure with two subplots
    fig, axs = plt.subplots(1, 1, figsize=(5, 6))

    axs.imshow(image_finetuned)
    axs.axis('off')

    # Add prompt and model name below the subplots
    fig.text(0.5, 0.01, 
             f"""
             Prompt: {args.prompt}
             Model: {args.model}
             Steps: {args.num_inference_steps}
             Seed: {seed}
             """, 
             ha='center'
    )

    # Save the figure
    output_path = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/individuals"
    current_datetime = datetime.now()
    # Format the date and time as HH:MM:SS DD/MM/YY
    filename = f"img_{current_datetime}_{args.prompt}.png"
    plt.savefig(f"{output_path}/metadata_{filename}")

    print(f"Image saved at {output_path}")


    negative_prompt = "teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant"

    seed = torch.random.initial_seed()

    generator = torch.manual_seed(seed)
    model_args = {
            "prompt": args.prompt,
            "image": image_finetuned,
            "control_image": image_finetuned,
            "strength": args.strength,
            "controlnet_conditioning_scale": 0.9,
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

    base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model,
        vae=vae,
        torch_dtype=torch.float16,
        controlnet=controlnet
    ).to("cuda")
    
    pipeline.safety_checker = None
    pipeline.enable_xformers_memory_efficient_attention()

    output = pipeline(**model_args).images[0]

    output.save(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/individuals/{filename}")


def parse_args():
    parser = argparse.ArgumentParser(description='Refine the generated image')
    parser.add_argument(
        '--model', 
        type=str, 
        help='Base model to generate the face',
        required=True
    )
    parser.add_argument(
        '--prompt', 
        type=str, 
        help='Prompt for image generation',
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
        type=float, 
        default=0.12, 
        help='Refinement strength'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    predict(args)
