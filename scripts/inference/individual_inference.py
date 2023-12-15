import argparse
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
from datetime import datetime


def plot_combined_images(model, prompt, num_inference_steps=50):
    seed = 2023
    generator = torch.manual_seed(seed)
    print("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained(f"./../../models/{model}", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None).to("cuda")
    print("Model loaded")

    image_finetuned = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]

    print("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-small", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None).to("cuda")
    print("Model loaded")

    image_vanilla = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]

    # Create a Matplotlib figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Display images in subplots
    axs[0].imshow(image_vanilla)
    axs[0].axis('off')
    axs[0].set_title('Vanilla Model')

    axs[1].imshow(image_finetuned)
    axs[1].axis('off')
    axs[1].set_title('Finetuned Model')

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
    output_path = "./../../output/images"
    current_datetime = datetime.now()
    # Format the date and time as HH:MM:SS DD/MM/YY
    filename = f"img_{current_datetime}.png"
    plt.savefig(f"{output_path}/{filename}")

    # Display the saved image file
    plt.show()

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
        default=50, 
        help='Number of inference steps'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    plot_combined_images(args.model, args.prompt, args.num_inference_steps)
