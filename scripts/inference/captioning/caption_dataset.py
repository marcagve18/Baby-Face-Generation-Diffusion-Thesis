import requests
import datasets
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline

def caption(data):
    prompt = f"USER: <image>\n{describe_prompt}\nASSISTANT:"
    outputs = pipe(data['image'], prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    outputs = list(map(lambda x: x[0]['generated_text'].split("ASSISTANT:")[1].strip(), outputs))
    return {"custom_caption": outputs}

dataset_name = "alexg99/captioned_flickr_faces"
dataset = datasets.load_dataset(dataset_name)

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

describe_prompt = """Describe the following face in detail. The output must be like if it was a prompt to generate the face. You must write it a specific structure. I'll give you some examples:
A smiling elderly Caucasian woman with gray hair posing frontally.
A neutral-faced baby with black hair in a lateral pose
A happy elderly Asian woman with white hair in a frontal pose.
A neutral-faced baby African boy with black hair in a frontal pose
A smiling middle-aged man with brown hair posing laterally.
A smiling middle-aged Asian man with red hair in a lateral pose.
"""


captioned_dataset = dataset['train'].map(caption, batched=True, batch_size=10)
captioned_dataset.save_to_disk(f"/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/{dataset_name.replace('/', '-')}-custom_caption")