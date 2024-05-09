from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionControlNetImg2ImgPipeline
import torch 
from PIL import Image

class Refiner():

    def __init__(self, dtype = torch.float16):
        self.controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1e_sd15_tile",
        torch_dtype=dtype
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema",
            torch_dtype=dtype
        )
        
        self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            vae=self.vae,
            torch_dtype=dtype,
            controlnet=self.controlnet
        ).to("cuda")
        self.pipeline.safety_checker = None
        self.pipeline.enable_xformers_memory_efficient_attention()

    def run(self, image : Image, prompt : str, strength=0.12, num_inference_steps=20) -> Image:
        negative_prompt = "teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant"

        seed = torch.random.initial_seed()

        generator = torch.manual_seed(seed)
        model_args = {
            "prompt": prompt,
            "image": image,
            "control_image": image,
            "strength": strength    ,
            "controlnet_conditioning_scale": 0.9,
            "negative_prompt": negative_prompt,
            "guidance_scale": 7,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "guess_mode": False,
        }

        return self.pipeline(**model_args).images[0]
