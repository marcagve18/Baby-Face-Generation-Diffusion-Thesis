import argparse
from diffusers import StableDiffusionPipeline,  EulerDiscreteScheduler
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os


def plot_combined_images(model, prompt, num_inference_steps):
    seed = torch.random.initial_seed()
    generator = torch.manual_seed(seed)
    vanilla = "runwayml/stable-diffusion-v1-5"

    print("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained(f"./../../models/{model}", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    print("Model loaded")
    scheduler = EulerDiscreteScheduler.from_pretrained(vanilla, subfolder="scheduler")


    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    pipeline.safety_checker = None


    image_finetuned = pipeline(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
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
             Prompt: {prompt}
             Model: {model}
             Steps: {num_inference_steps}
             Seed: {seed}
             """, 
             ha='center'
    )

    # Save the figure
    output_path = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/refiner-comparison/no-refiner"
    current_datetime = datetime.now()
    # Format the date and time as HH:MM:SS DD/MM/YY
    filename = f"img_{current_datetime}_{prompt}.png"
    plt.savefig(f"{output_path}/metadata_{filename}")
    image_finetuned.save(f"{output_path}/{filename}")

    print(f"Image saved at {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and save a combined image from two models based on a prompt.')
    parser.add_argument(
        '--model', 
        type=str, 
        help='Model name',
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    plot_combined_images(args.model, args.prompt, args.num_inference_steps)
