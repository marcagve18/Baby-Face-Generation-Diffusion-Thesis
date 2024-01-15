import datasets
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline

class Captioner:
    def __init__(self, model_id, describe_prompt):
        self.describe_prompt = describe_prompt
        
        # Load the dataset

        # Load the quantization configuration
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load the pipeline
        self.pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": self.quantization_config})
    
    def caption(self, image):

        # Create prompt
        prompt = f"USER: <image>\n{self.describe_prompt}\nASSISTANT:"
        
        # Get output
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        
        # Process output
        caption = output[0]['generated_text'].split("ASSISTANT:")[1].strip()
        
        return caption

class LlavaCaptioner(Captioner):
    def __init__(self, describe_prompt):
        model_id = "llava-hf/llava-1.5-7b-hf"
        super().__init__(model_id, describe_prompt)
