import torch
import matplotlib.pyplot as plt
import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
from PIL import Image

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
torch.cuda.empty_cache()

quantization_config = BitsAndBytesConfig(
    # load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )

model_id = "Ertugrul/Pixtral-12B-Captioner-Relaxed"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained(model_id)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
model.resize_token_embeddings(len(processor.tokenizer))

conversation = [
    {
        "role": "user",
        "content": [
            
            {"type": "text", "text": "Describe the image.\n"},
            {
                "type": "image",
            }
        ],
    }
]

PROMPT = processor.apply_chat_template(conversation, add_generation_prompt=True)

class DataCollator:
    def __init__(self, processor):
        self.processor = processor
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __call__(self, examples):
        example = examples[0]

        image = example["image"]

        user_prompt = "Describe this image"
        answer = ",".join(example["clip_tags_ViT_L_14"])

        PROMPT = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        answer = f"{answer}<|end|>\n<|endoftext|>"

        # Tokenize input and ensure uniform padding
        batch = self.processor(PROMPT, [image], return_tensors="pt", padding="max_length", max_length=256)

        # Tokenize answer with consistent max length
        answer_input_ids = self.processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt", padding="max_length", max_length=256)["input_ids"]

        # Concatenate prompt and answer tokens
        concatenated_input_ids = torch.cat([batch["input_ids"], answer_input_ids], dim=1)

        # Generate labels, masking the prompt part
        ignore_index = -100
        labels = torch.cat(
            [
                torch.full_like(batch["input_ids"], ignore_index),  # Mask the prompt
                answer_input_ids,  # Keep the answer as label
            ],
            dim=1,
        )

        # Check the size of input_ids after concatenation
        max_length_after_concat = concatenated_input_ids.shape[1]
        attention_mask = torch.cat(
            [batch["attention_mask"], torch.ones_like(answer_input_ids)], dim=1
        ).to(concatenated_input_ids.device)

        # Ensure all tensors have consistent shapes
        concatenated_input_ids = concatenated_input_ids[:, :max_length_after_concat]
        attention_mask = attention_mask[:, :max_length_after_concat]
        labels = labels[:, :max_length_after_concat]
        print(labels)

        batch["input_ids"] = concatenated_input_ids
        batch["labels"] = labels
        batch["attention_mask"] = attention_mask

        # Ensure only floating-point tensors require gradients
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        #         batch[key] = value.clone().detach().requires_grad_(True)
        return batch

data_collator = DataCollator(processor)

image = Image.open(r"/home/admin/test_data/image/test_img_!.jpg")

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


# you can try different resolutions or disable it completely
image = resize_image(image, 256)

inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")


# with torch.no_grad():
#     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#         generate_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.01, use_cache=True, top_k=20)
# output_text = processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

from datasets import load_dataset
train_dataset = load_dataset("Multimodal-Fatima/FGVC_Aircraft_train")
train_dataset = train_dataset['train'].select(range(10))
eval_dataset = load_dataset("Multimodal-Fatima/FGVC_Aircraft_test")

from peft import LoraConfig
 
# LoRA config based on QLoRA paper & Sebastian Raschka experiment
lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        use_rslora=True,
        target_modules=[
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.qkv_proj.weight",
        "self_attn.out_proj.weight",
        "mlp.gate_up_proj",
        "mlp.down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
)

from peft import get_peft_model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer
epochs = 1
lr = 3e-5
schedule = "constant"
run_name = f"finetune_geo_epoch_{epochs}_schedule_{schedule}"

training_args = TrainingArguments(
    num_train_epochs=epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=lr,
    weight_decay=0.01,
    logging_steps=0.1,
    output_dir="/home/admin/",
    evaluation_strategy="steps",
    eval_steps=0.2,
    lr_scheduler_type=schedule,
    fp16=True,
    remove_unused_columns=False,
    report_to="tensorboard",
    run_name=run_name,
    logging_dir=f"./logs/{run_name}"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset['test']
)

trainer.train()
trainer.save_model("/home/admin/checkpoint-1/")
processor.save_pretrained("/home/admin/checkpoint-1/")

def load_with_peft_adapter(base_model_id, model_path, device="cpu", dtype=torch.bfloat16):
    """
        load model via Huggingface AutoModelForCausalLM and AutoProcessor
    """
    torch.set_default_device(device)

    with torch.device(device):

        processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        model = LlavaForConditionalGeneration.from_pretrained(
            base_model_id,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=True
        ).eval()

        model.resize_token_embeddings(len(processor.tokenizer))

        finetuned_model = LlavaForConditionalGeneration.from_pretrained(
            model,
            model_path
        ).eval()
        finetuned_model.to(device)
        finetuned_model.config.use_cache = True

        return finetuned_model, processor

finetuned_model, processor = load_with_peft_adapter(model_id, "/home/admin/checkpoint-1/", device="cuda", dtype=torch.bfloat16)

def inference_with_finetuned_model(id, dataset, finetuned_model, processor, tokenizer, device="cuda", temperature = 0.0):
  image = dataset[id]["image"].convert("RGB")

  userPrompt = "Describe this image"

  prompt = f"<|user|>\n<|image_1|>\n{userPrompt}<|end|>\n<|assistant|>\n"

  inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

  generation_args = {
      "max_new_tokens": 256,
      "temperature": 0.01
  }

  with torch.no_grad():
    generate_ids = finetuned_model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

  generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

  decoded_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
  response = decoded_output[0]

  return response

finetuned_model_output = [inference_with_finetuned_model(i, test_dataset, finetuned_model, processor, processor.tokenizer) for i in range(5)]
print(finetuned_model_output)
