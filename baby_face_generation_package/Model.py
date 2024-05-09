from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image
from .Refiner import Refiner
from dataclasses import dataclass

@dataclass
class GenerationOutput:
    prompt:str
    seed: str
    image: Image


class BabyGenerator:

    def __init__(self, dtype = torch.float16):
        self.pipeline = StableDiffusionPipeline.from_pretrained("marcagve18/baby-face-generation", torch_dtype=dtype, use_safetensors=True, safety_checker=None).to("cuda")
        self.pipeline.safety_checker = None
        self.scheduler = EulerDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.refiner = Refiner()

    def generate(self, prompt, num_inference_steps = 20, seed = None) -> GenerationOutput:
        negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        
        if seed is None:
            seed = torch.random.initial_seed()

        generator = torch.manual_seed(seed)

        non_refined_image = self.pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=5,
            scheduler=self.scheduler, 
            generator=generator
        ).images[0]
        
        refined = self.refiner.run(
            non_refined_image,
            prompt,
        )

        return GenerationOutput(prompt, seed, refined)
    
    def edit(self, input : GenerationOutput, prompt : str, num_inference_steps=20, switch_step=4) -> GenerationOutput:
        negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        
        generator = torch.manual_seed(input.seed)
        image_edited = self.pipeline.call_v2(
            prompt_1=input.prompt, 
            prompt_2=prompt, 
            switch_step=switch_step,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=5,
            scheduler=self.scheduler, 
            generator=generator,
        ).images[0]

        refined = self.refiner.run(
            image_edited,
            prompt,
        )

        return GenerationOutput(prompt, input.seed, refined)


        
