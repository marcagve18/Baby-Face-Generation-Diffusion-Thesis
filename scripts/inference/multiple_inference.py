import argparse
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
from datetime import datetime

def plot_combined_images(models, prompt, num_inference_steps, seed):
    if seed is None:
        seed = torch.random.initial_seed()
    generator = torch.manual_seed(seed)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    num_models = len(models) + 1

    fig_width = 500 * px * num_models  # Set a base width for each model
    fig, axs = plt.subplots(1, num_models, figsize=(fig_width, 600*px))

    # Load and plot the vanilla model first
    print(f"Loading vanilla model")
    pipeline = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-small", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None).to("cuda")
    print("Vanilla model loaded")

    image = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]

    # Display the vanilla model image and name
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title('Vanilla Model')

    # Load and plot the rest of the models
    for i, model in enumerate(models):
        print(f"Loading model {model}")
        pipeline = StableDiffusionPipeline.from_pretrained(f"./../../models/{model}", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None).to("cuda")
        print(f"Model {model} loaded")

        image = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]

        # Display images in subplots
        axs[i+1].imshow(image)
        axs[i+1].axis('off')
        axs[i+1].set_title(f'{model}')

    # Add prompt and model name below the subplots
    fig.text(0.5, 0, 
             f"""
             Prompt: {prompt}
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

    print(f"Image saved at {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and save a combined image from multiple models based on a prompt.')
    parser.add_argument(
        '--models', 
        type=str, 
        nargs='+',
        help='List of model names',
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
        '--seed', 
        type=int, 
        default=None, 
        help='Number of inference steps'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    plot_combined_images(args.models, args.prompt, args.num_inference_steps, args.seed)
