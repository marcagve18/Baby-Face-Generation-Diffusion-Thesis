import datasets
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline

class Captioner:
    def __init__(self, model_id, dataset_name, describe_prompt):
        self.describe_prompt = describe_prompt
        
        # Load the dataset
        self.dataset = datasets.load_dataset(dataset_name)

        # Load the quantization configuration
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load the pipeline
        self.pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": self.quantization_config})

    def caption(self, image):
        # Prepare the data
        data = {'image': image}

        # Create prompt
        prompt = f"USER: <image>\n{self.describe_prompt}\nASSISTANT:"
        
        # Get outputs
        outputs = self.pipe(data['image'], prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        
        # Process outputs
        captions = list(map(lambda x: x[0]['generated_text'].split("ASSISTANT:")[1].strip(), outputs))
        
        return {"custom_caption": captions}

class LlavaCaptioner(Captioner):
    def __init__(self, dataset_name, describe_prompt):
        model_id = "llava-hf/llava-1.5-7b-hf"
        super().__init__(model_id, dataset_name, describe_prompt)
