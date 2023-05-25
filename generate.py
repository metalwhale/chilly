import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

BASE_MODEL = "decapoda-research/llama-7b-hf"
MODELS_DIR = "data/models"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
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
    prompt = input("prompt: ")
    if prompt == "exit":
        break
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    outputs = tokenizer.batch_decode(model.generate(input_ids=input_ids, max_new_tokens=128))
    print(outputs[0])
