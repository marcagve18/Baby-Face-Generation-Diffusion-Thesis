import datasets
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline
import os


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

print("Loading model")
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
print("Model loaded")


def caption(data):
    describe_prompt = """
    Describe the image as if it was a prompt for generating. Follow this format:

    [pose(Profile, Frontal)] portrait of a [expression (smiling, neutral, crying, sad)] [ethnicity] [baby/man/woman]

    Some examples:
    Frontal portrait of a smiling black man
    """

    prompt = f"USER: <image>\n{describe_prompt}\nASSISTANT:"
    outputs = pipe(data['image'], prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    outputs = list(map(lambda x: x[0]['generated_text'].split("ASSISTANT:")[1].strip(), outputs))
    print(f"Outputs: {outputs}")

    return {"custom_caption": outputs}

def run():

    print("Loading dataset")
    dataset_name = "alexg99/captioned_flickr_faces"
    dataset = datasets.load_dataset(dataset_name)
    print("Dataset loaded")

    dataset = dataset.remove_columns("conditioning_image")
    describe_prompt = """
        Describe the image as if it was a prompt for generating. Follow this format:

        [pose(Profile, Frontal)] portrait of a [expression (smiling, neutral, crying, sad)] [ethnicity] [baby/man/woman]

        Some examples:
        Frontal portrait of a smiling black man
        """

    captioned_dataset = dataset['train'].map(caption, batched=True, batch_size=100)

    captioned_dataset.save_to_disk(f"{os.environ.get('HOME_PATH')}/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/{dataset_name.replace('/', '-')}-custom_caption_v2")

if __name__ == '__main__':
    run()