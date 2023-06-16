import sys

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

MODELS_DIR = "data/models"

base_model = "decapoda-research/llama-7b-hf"
if len(sys.argv) >= 2:
    base_model = sys.argv[1]

tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    MODELS_DIR,
    torch_dtype=torch.float16,
)
while True:
    message = input("message: ")
    if message == "exit":
        break
    message = f"This is a short chat between friends in Vietnamese:\n- {message}.\n- "
    input_ids = tokenizer(message, return_tensors="pt")["input_ids"].to("cuda")
    outputs = tokenizer.batch_decode(model.generate(input_ids=input_ids, max_new_tokens=128))
    print(outputs[0])
