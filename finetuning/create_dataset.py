from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
import torch
import matplotlib.pyplot as plt
import json
import glob

files = glob.glob('/home/admin/dataset-ninja/floodnet-2021:-track-2/train_image/img/*.JPG')
quantization_config = BitsAndBytesConfig(
    # load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )

model_id = "Ertugrul/Pixtral-12B-Captioner-Relaxed"
model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

# for quantization just use this instead of previous load
# model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

def resize_image(image, target_size=768):
    """Resize the image to have the target size on the shortest side."""
    width, height = image.size
    if width < height:
        new_width = target_size
        new_height = int(height * (new_width / width))
    else:
        new_height = target_size
        new_width = int(width * (new_height / height))
    return image.resize((new_width, new_height), Image.LANCZOS)

conversation = [
    {
        "role": "user",
        "content": [
            
            {"type": "text", "text": "Describe the image in terms of the object and features, their qualities, quantities and spatial significance.\n"},
            {
                "type": "image",
            }
        ],
    }
]

PROMPT = processor.apply_chat_template(conversation, add_generation_prompt=True)
data = []
for file in files:
    image = Image.open(file)
    image = resize_image(image, 256)
    inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generate_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.01, use_cache=True, top_k=20)
    output_text = processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    data.append(
            {   
                "text": {output_text},
                "image": {file}
            })

def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets(i) for i in obj]
    else:
        return obj

data_ser = convert_sets(data)
with open('/home/admin/dataset.json', 'w') as json_file:
    json.dump(data_ser, json_file, indent=4)
