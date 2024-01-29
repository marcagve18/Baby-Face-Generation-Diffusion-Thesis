import sys
import os
captioner_path = os.environ['HOME_PATH'] + "/TFG/Baby-Face-Generation-Diffusion-Thesis/scripts/datasets/captioning"
sys.path.append(captioner_path)
from Captioner import LlavaCaptioner
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import AutoModel
from numpy.linalg import norm
from PIL import Image
import json
import time
import matplotlib.pyplot as plt


describe_prompt = """
Describe the image as if it was a prompt for generating. Follow this format:

[pose(Profile, Frontal)] portrait of a [expression (smiling, neutral, crying, sad)] [ethnicity] [baby/man/woman]

Some examples:
Frontal portrait of a smiling black man
"""

captioner = LlavaCaptioner(describe_prompt)
embedder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

evaluation_prompts = ["Frontal portrait of a smiling white baby"]
model = "sd_15_v5"

seed = torch.random.initial_seed()
generator = torch.manual_seed(seed)
checkpoints = range(2000, 120000, 2000)
evaluation = {}
num_inference_steps = 20
scheduler = EulerDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")



for step, checkpoint in enumerate(checkpoints):
    print(f"Output step: {step+1}/{len(checkpoints)}")
    unet = UNet2DConditionModel.from_pretrained(
        f"{os.environ['HOME_PATH']}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/{model}/checkpoint-{checkpoint}/unet", 
        torch_dtype=torch.float16, 
        subfolder="unet", 
        use_safetensors=True
    ).to("cuda")

    start = time.time()
    pipeline = StableDiffusionPipeline.from_pretrained(
        f"{os.environ['HOME_PATH']}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/{model}",
        unet= unet,
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        safety_checker=None).to("cuda")
    print(f"Model loaded in {time.time()-start}s")

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    evaluation[checkpoint] = {
        "images" : [],
        "scores" : []
    }

    for prompt in evaluation_prompts:
        generator = torch.manual_seed(seed)
        image_finetuned = pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, 
            guidance_scale=5,
            scheduler=scheduler, 
            generator=generator).images[0]
        
        # caption = captioner.caption(image_finetuned)
        # embeddings = embedder.encode([prompt, caption])
        evaluation[checkpoint]["images"].append(image_finetuned)
        #evaluation[checkpoint]["scores"].append(cos_sim(embeddings[0], embeddings[1]))
        evaluation[checkpoint]["scores"].append(-1)


    del pipeline

eval_path = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/evaluations/{model}"
if not os.path.exists(eval_path):
    os.mkdir(eval_path)

evaluation_output = {}
for key in evaluation.keys():
    evaluation_output["metadata"] = {
        "seed" : seed,
        "steps" : num_inference_steps
    }
    evaluation_output[key] = {
        "scores" : [float(i) for i in evaluation[key]["scores"]]
    }

with open(f"{eval_path}/output.json", "w") as fp:
    json.dump(evaluation_output , fp) 


num_images = len(next(iter(evaluation.values()))['images'])

def add_text_to_image(image, text):
    """ Add text below an image using Matplotlib """
    plt.imshow(image)
    plt.axis('off')
    plt.text(0, image.size[1], text, fontsize=12, color='white', backgroundcolor='black')
    plt.tight_layout(pad=0)
    
    # Save the image with text as a temporary file
    temp_filename = f'temp_{text}.png'
    plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return Image.open(temp_filename), temp_filename

for i in range(num_images):
    images_for_gif = [entry['images'][i] for entry in evaluation.values()]

    temp_filenames = []
    modified_images = []
    for idx, img in enumerate(images_for_gif):
        img_with_text, filename = add_text_to_image(img, f"{checkpoints[idx]}_{i}")
        temp_filenames.append(filename)
        img_with_text = img_with_text.convert(images_for_gif[0].mode).resize(images_for_gif[0].size)
        modified_images.append(img_with_text)

    # Save the GIF
    modified_images[0].save(f'{eval_path}/output_{i}.gif', save_all=True, append_images=modified_images[1:], duration=500, loop=0)

    # Optionally, delete the temporary files
    for img in modified_images:
        img.close()
    for fname in temp_filenames:
        os.remove(fname)
    

