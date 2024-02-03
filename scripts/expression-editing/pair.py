import argparse
from diffusers import EulerDiscreteScheduler
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import imageio
import os

intermediate_images = {
    "neutral" : [],
    "edited" : []
}
output_mode = ""

def decode_latents(pipe, step_index, timestep, image, callback_kwargs):
    global output_mode
    intermediate_images[output_mode].append(image[0])
    return callback_kwargs

def plot_combined_images(model, prompt1, prompt2, num_inference_steps, switch_step):
    global output_mode 

    seed = torch.random.initial_seed()
    generator = torch.manual_seed(seed)
    vanilla = "runwayml/stable-diffusion-v1-5"
    scheduler = EulerDiscreteScheduler.from_pretrained(vanilla, subfolder="scheduler")

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    print("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained(f"./../../models/{model}", use_safetensors=True, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    print("Model loaded")

    generator = torch.manual_seed(seed)
    output_mode = "neutral"
    image_neutral = pipeline(
        prompt=prompt1, 
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=5,
        scheduler=scheduler, 
        generator=generator,
        callback_on_step_end=decode_latents
        ).images[0]
    
    output_mode = "edited"
    generator = torch.manual_seed(seed)
    image_edited = pipeline.call_v2(
        prompt_1=prompt1, 
        prompt_2=prompt2, 
        switch_step=switch_step,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=5,
        scheduler=scheduler, 
        generator=generator,
        callback_on_step_end=decode_latents
        ).images[0]

    # Create a Matplotlib figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 7))

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
             Prompt_1: {prompt1}
             Prompt_2: {prompt2}
             Model: {model}
             Switch step: {switch_step}
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

    intermediate_image_paths = []
    for i in range(num_inference_steps):
        # Create a Matplotlib figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 7))

        # Display images in subplots
        axs[0].imshow(intermediate_images['neutral'][i])
        axs[0].axis('off')
        axs[0].set_title('Neutral Image')

        axs[1].imshow(intermediate_images['edited'][i])
        axs[1].axis('off')
        axs[1].set_title('Edited Image')

        # Add prompt and model name below the subplots
        fig.text(0.5, 0.01, 
                f"""
                Step: {i+1}
                Prompt_1: {prompt1}
                Prompt_2: {prompt2}
                Model: {model}
                Switch step: {switch_step}
                Steps: {num_inference_steps}
                Seed: {seed}
                """, 
                ha='center'
        )
        # Save each figure as an image
        output_path = f"./../../output/images/expression-edition/intermediate_step_{i}.png"
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory

        # Append the path to the list
        intermediate_image_paths.append(output_path)

    # Read images from intermediate steps and create a GIF
    images = []
    for path in intermediate_image_paths:
        images.append(imageio.v2.imread(path))

    # Save the GIF
    output_gif_path = f"./../../output/images/expression-edition/intermediate_images_{current_datetime}.gif"
    imageio.mimsave(output_gif_path, images)

    for i in range(num_inference_steps):
        os.remove(intermediate_image_paths[i])

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
        '--prompt1', 
        type=str, 
        help='Prompt for image generation',
        required=True
    )
    parser.add_argument(
        '--prompt2', 
        type=str, 
        help='Prompt for image modification',
        required=True
    )
    parser.add_argument(
        '--num_inference_steps', 
        type=int, 
        default=20, 
        help='Number of inference steps'
    )
    parser.add_argument(
        '--switch_step', 
        type=int, 
        help='Step number in which prompts will be switched'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    plot_combined_images(args.model, args.prompt1, args.prompt2, args.num_inference_steps, args.switch_step)
