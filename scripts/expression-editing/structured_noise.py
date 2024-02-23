import argparse
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import numpy as np
import torchvision.transforms as transforms 

intermediate_images = {
    "neutral" : [],
    "edited" : []
}
output_mode = ""

def resize_to_512(original_image):
    """
    Resizes an image to 512x512 by adding blank margins or downsizing.
    
    Parameters:
    - original_image: PIL.Image, the original image to resize.
    
    Returns:
    - PIL.Image object resized to 512x512.
    """
    original_size = original_image.size
    target_size = (512, 512)
    
    # Calculate the ratio of the target dimensions to the original dimensions
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    
    # If the image is larger than the target, downsize it first
    if ratio < 1:
        new_size = tuple([int(x * ratio) for x in original_size])
        resized_image = original_image.resize(new_size, Image.ANTIALIAS)
    else:
        resized_image = original_image
    
    # Create a new image with white background
    new_image = Image.new("RGB", target_size, "white")
    
    # Calculate top-left position to paste the resized image
    top_left = ((target_size[0] - resized_image.size[0]) // 2, (target_size[1] - resized_image.size[1]) // 2)
    
    # Paste the resized image onto the new image
    new_image.paste(resized_image, top_left)
    
    return new_image

def revert_resize(processed_image, original_size):
    """
    Attempts to revert a resized 512x512 image back to its original dimensions.
    This function assumes the original dimensions are known or passed explicitly.
    
    Parameters:
    - processed_image: PIL.Image, the processed image to revert.
    - original_size: tuple, the original dimensions of the image (width, height).
    
    Returns:
    - PIL.Image object reverted to its original size.
    """
    # Calculate the area of the original image within the 512x512 canvas
    start_x = (512 - original_size[0]) // 2
    start_y = (512 - original_size[1]) // 2
    end_x = start_x + original_size[0]
    end_y = start_y + original_size[1]
    
    # Crop the image back to its original dimensions
    original_image = processed_image.crop((start_x, start_y, end_x, end_y))
    
    return original_image

def decode_latents(pipe, step_index, timestep, image, callback_kwargs):
    global output_mode
    intermediate_images[output_mode].append(image[0])
    return callback_kwargs

def add_gaussian_noise(image_path, noise_scale):
    
    return noisy_image    

def plot_combined_images(model, prompt, image_path, num_inference_steps):
    global output_mode 

    seed = torch.random.initial_seed()
    generator = torch.manual_seed(seed)
    vanilla = "runwayml/stable-diffusion-v1-5"
    scheduler = EulerDiscreteScheduler.from_pretrained(vanilla, subfolder="scheduler")

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    image = resize_to_512(Image.open(image_path))

    print("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained(f"./../../models/{model}", use_safetensors=True, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    print("Model loaded")

    # Define a transform to convert PIL  
    # image to a Torch tensor 
    transform = transforms.Compose([ 
        transforms.PILToTensor() 
    ]) 
    
    # transform = transforms.PILToTensor() 
    # Convert the PIL image to Torch tensor 
    img_tensor = transform(image).unsqueeze(0)
    print(img_tensor.shape)
    latents = pipeline.vae.encode(img_tensor.half().to('cuda')).latent_dist.sample() * pipeline.vae.config.scaling_factor
    noise = torch.randn_like(latents)

    print("Noisy latents encoded")

    generator = torch.manual_seed(seed)
    output_mode = "neutral"
    image_neutral = pipeline(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=5,
        scheduler=scheduler, 
        generator=generator,
        callback_on_step_end=decode_latents
        ).images[0]
    print("Vanilla image computed")
    
    output_mode = "edited"
    generator = torch.manual_seed(seed)
    image_edited = pipeline(
        latents=latents,
        prompt=prompt, 
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
             Prompt_1: {prompt}
             Model: {model}
             Steps: {num_inference_steps}
             Seed: {seed}
             """, 
             ha='center'
    )

    # Save the figure
    output_path = "./../../output/images/expression-edition/structured-noise"
    current_datetime = datetime.now()
    # Format the date and time as HH:MM:SS DD/MM/YY
    filename = f"img_{current_datetime}.png"
    plt.savefig(f"{output_path}/{filename}")    

    print(f"Image saved")

def parse_args():
    parser = argparse.ArgumentParser(description='Bring an image and edit its expression')
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
        '--image_path', 
        type=str, 
        help='Path of the initial image',
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
    plot_combined_images(args.model, args.prompt, args.image_path, args.num_inference_steps)
