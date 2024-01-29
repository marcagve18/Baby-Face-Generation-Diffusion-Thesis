import argparse
from diffusers import EulerDiscreteScheduler
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
from datetime import datetime




def plot_combined_images(model, prompt, num_inference_steps):
    seed = torch.random.initial_seed()
    generator = torch.manual_seed(seed)
    vanilla = "runwayml/stable-diffusion-v1-5"
    scheduler = EulerDiscreteScheduler.from_pretrained(vanilla, subfolder="scheduler")

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    print("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained(f"./../../models/{model}", use_safetensors=True, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    print("Model loaded")

    generator = torch.manual_seed(seed)
    image_neutral = pipeline(
        prompt="Frontal portrait of a smiling white baby", 
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=5,
        scheduler=scheduler, 
        generator=generator).images[0]
    
    generator = torch.manual_seed(seed)
    image_edited = pipeline.call_v2(
        prompt_1="Frontal portrait of a smiling white baby", 
        prompt_2="Frontal portrait of a sad white baby", 
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=5,
        scheduler=scheduler, 
        generator=generator).images[0]


    # Create a Matplotlib figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Display images in subplots
    axs[0].imshow(image_neutral)
    axs[0].axis('off')
    axs[0].set_title('Neutral Image')

    axs[1].imshow(image_edited)
    axs[1].axis('off')
    axs[1].set_title('Edited Image')

    # Add prompt and model name below the subplots
    fig.text(0.5, 0.01, 
             f"""
             Vanilla: {vanilla}
             Prompt: {prompt}
             Model: {model}
             Steps: {num_inference_steps}
             Seed: {seed}
             """, 
             ha='center'
    )

    # Save the figure
    output_path = "./../../output/images/expression-edition"
    current_datetime = datetime.now()
    # Format the date and time as HH:MM:SS DD/MM/YY
    filename = f"img_{current_datetime}.png"
    plt.savefig(f"{output_path}/{filename}")

    # Display the saved image file
    plt.show()

    print(f"Image saved at {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate an image and edit its expression')
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
