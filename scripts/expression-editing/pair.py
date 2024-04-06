import argparse
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL
)
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import imageio
import os
from PIL import Image

intermediate_images = {
    "neutral" : [],
    "edited" : []
}
output_mode = ""

def refine(img, prompt) -> Image : 
    negative_prompt = "teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant"

    seed = torch.random.initial_seed()

    generator = torch.manual_seed(seed)
    model_args = {
            "prompt": prompt,
            "image": img,
            "control_image": img,
            "strength": 0.12,
            "controlnet_conditioning_scale": 0.9,
            "negative_prompt": negative_prompt,
            "guidance_scale": 7,
            "generator": generator,
            "num_inference_steps": 20,
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

    return pipeline(**model_args).images[0]

def decode_latents(pipe, step_index, timestep, image, callback_kwargs):
    global output_mode
    intermediate_images[output_mode].append(image[0])
    return callback_kwargs

def plot_combined_images(model, prompt1, prompt2, num_inference_steps, switch_step):
    global output_mode 

    seed = 16355036249119675404 #torch.random.initial_seed()
    vanilla = "runwayml/stable-diffusion-v1-5"
    scheduler = EulerDiscreteScheduler.from_pretrained(vanilla, subfolder="scheduler")

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    print("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/{model}", use_safetensors=True, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    print("Model loaded")

    # Save the figure
    output_path = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/expression/prompt-swapping"
    current_datetime = datetime.now()
    # Format the date and time as HH:MM:SS DD/MM/YY
    filename = f"img_{current_datetime}.png"

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
    
    neutral_refined = refine(image_neutral, prompt1)
    neutral_refined.save(f"{output_path}/original_{filename}")

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

    edited_refined = refine(image_edited, prompt2)
    edited_refined.save(f"{output_path}/edited{filename}")

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

    # Save fig
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
        output_path_img = f"{output_path}/intermediate_step_{i}.png"
        plt.savefig(output_path_img)
        plt.close()  # Close the figure to free memory

        # Append the path to the list
        intermediate_image_paths.append(output_path_img)

    # Read images from intermediate steps and create a GIF
    images = []
    for path in intermediate_image_paths:
        images.append(imageio.v2.imread(path))

    # Save the GIF
    output_gif_path = f"{output_path}/intermediate_images_{current_datetime}.gif"
    imageio.mimsave(output_gif_path, images)

    #for i in range(num_inference_steps):
     #   os.remove(intermediate_image_paths[i])

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
