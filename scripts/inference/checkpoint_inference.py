import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import time
import matplotlib.pyplot as plt
from datetime import datetime


def inference(model, prompt, num_inference_steps, checkpoint):
    seed = torch.random.initial_seed()
    generator = torch.manual_seed(seed)

        
    unet = UNet2DConditionModel.from_pretrained(
        f"{os.environ['HOME_PATH']}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/{model}/checkpoint-{checkpoint}/unet", 
        torch_dtype=torch.float16, 
        subfolder="unet", 
        use_safetensors=True
    ).to("cuda")

    start = time.time()
    pipeline = StableDiffusionPipeline.from_pretrained(
        f"runwayml/stable-diffusion-v1-5",
        unet= unet,
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        safety_checker=None).to("cuda")
    print(f"Model loaded in {time.time()-start}s")

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"


    image_finetuned = pipeline(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps, 
        generator=generator).images[0]
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))

    # Display image in the subplot
    axs.imshow(image_finetuned)
    axs.axis('off')
    axs.set_title(f'Checkpoint {checkpoint}')

    # Add prompt and model name below the subplot
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
    output_path = "./../../output/images"
    current_datetime = datetime.now()
    # Format the date and time as HH:MM:SS DD/MM/YY
    filename = f"img_{current_datetime}.png"
    plt.savefig(f"{output_path}/{filename}")
    
    

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
        default=50, 
        help='Number of inference steps'
    )
    parser.add_argument(
        '--checkpoint', 
        type=int, 
        help='Checkpoint number',
        required=True
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(args.model, args.prompt, args.num_inference_steps, args.checkpoint)
    

