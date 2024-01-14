import sys
import os
captioner_path = os.environ['HOME_PATH'] + "/TFG/Baby-Face-Generation-Diffusion-Thesis/scripts/datasets/captioning"
sys.path.append(captioner_path)
from Captioner import LlavaCaptioner
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import AutoModel
from numpy.linalg import norm
from PIL import Image
import json
import time



describe_prompt = """
Describe the image as if it was a prompt for generating. Follow this format:

[pose(Profile, Frontal)] portrait of a [expression (smiling, neutral, crying, sad)] [ethnicity] [baby/man/woman]

Some examples:
Frontal portrait of a smiling black man
"""

captioner = LlavaCaptioner(describe_prompt)
embedder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

evaluation_prompts = ["Frontal portrait of a smiling white baby"]
model = "sd_15_v3"

seed = torch.random.initial_seed()
generator = torch.manual_seed(seed)
checkpoints = range(500, 2000, 500)
evaluation = {}

for checkpoint in checkpoints:
    unet = UNet2DConditionModel.from_pretrained(
        f"{os.environ['HOME_PATH']}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/{model}/checkpoint-{checkpoint}/unet", 
        torch_dtype=torch.float16, 
        subfolder="unet", 
        use_safetensors=True
    ).to("cuda")

    print("Loading model")
    start = time.time()
    pipeline = StableDiffusionPipeline.from_pretrained(
        f"{os.environ['HOME_PATH']}/TFG/Baby-Face-Generation-Diffusion-Thesis/models/{model}",
        unet= unet,
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        safety_checker=None).to("cuda")
    end = time.time()
    print(f"Model loaded in {end-start}s")

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    num_inference_steps = 50
    evaluation[checkpoint] = {
        "images" : [],
        "scores" : []
    }

    for prompt in evaluation_prompts:
        image_finetuned = pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, 
            generator=generator).images[0]
        
        caption = captioner.caption(image_finetuned)
        embeddings = embedder.encode([prompt, caption])
        print(f"Generator prompt: {prompt}\nCaptioned prompt: {captioner.caption(image_finetuned)}\nSimilarity: {cos_sim(embeddings[0], embeddings[1])}")
        evaluation[checkpoint]["images"].append(image_finetuned)
        evaluation[checkpoint]["scores"].append(cos_sim(embeddings[0], embeddings[1]))


eval_path = f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/output/evaluations/{model}"
os.mkdir(eval_path)

evaluation_output = {}
for key in evaluation.keys():
    evaluation_output[key] = {
        "scores" : evaluation[key]["scores"] 
    }

with open(f"{eval_path}/output.json", "w") as fp:
    json.dump(evaluation_output , fp) 


num_images = len(next(iter(evaluation.values()))['images'])

for i in range(num_images):
    images_for_gif = [entry['images'][i] for entry in evaluation.values()]

    # Ensure all images are in the same mode and size
    images_for_gif = [img.convert(images_for_gif[0].mode).resize(images_for_gif[0].size) for img in images_for_gif]

    # Save the GIF
    images_for_gif[0].save(f'{eval_path}/output_{i}.gif', save_all=True, append_images=images_for_gif[1:], duration=500, loop=0)
        

