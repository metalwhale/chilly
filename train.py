BASE_MODEL = "decapoda-research/llama-7b-hf"
ZIP_FILE = "data/slack-export-Mart6-2018-Mar31-2023.zip"
RAW_DATA_DIR = "data/raw_data"
TRAIN_FILE = "data/train.json"
VAL_FILE = "data/val.json"
OUTPUT_DIR = "data/output"
TRAIN_LENGTH = 16000
VAL_LENGTH = 2000
EPOCHS = 4

import shutil
import zipfile

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
shutil.rmtree(RAW_DATA_DIR, ignore_errors=True)
with zipfile.ZipFile(ZIP_FILE, "r") as zip_file:
    zip_file.extractall(RAW_DATA_DIR)

import os
import glob
import json
import re
import random
from collections import defaultdict
from typing import List

class Message:
    text: str
    ts: str  # timestamp
    user: str
    def __init__(self, text: str, ts: str, user: str) -> "Message":
        self.text = text
        self.ts = ts
        self.user = user

class Conversation:
    messages: List[Message]
    def __init__(self) -> "Conversation":
        self.messages = []

def generate_dataset(input_dir: str):
    CONVERSATION_DISTANCE_TIME = 15 * 60  # in seconds
    CONVERSATION_MIN_LENGTH = 2
    conversations: list[Conversation] = []
    for channel in next(os.walk(input_dir))[1]:  # List all subdirectories
        thread_dict = defaultdict(list)
        for json_file_path in glob.glob(os.path.join(input_dir, channel, "*.json")):
            with open(json_file_path) as json_file:
                for message_obj in json.load(json_file):
                    if "thread_ts" in message_obj:  # Messages in a thread
                        if message_obj["thread_ts"] == message_obj["ts"]:  # Root message of the thread
                            thread_dict["main"].append(message_obj)
                        thread_dict[message_obj["thread_ts"]].append(message_obj)
                    elif "subtype" not in message_obj:  # Normal messages. See: https://api.slack.com/events/message#subtypes.
                        thread_dict["main"].append(message_obj)
        conversation = Conversation()
        for thread, message_obj_list in thread_dict.items():
            for message_obj in message_obj_list:
                if "user" not in message_obj or len(message_obj["text"]) == 0:
                    continue
                text = re.sub(r"(\n)+", ". ", message_obj["text"])
                last_message = None if len(conversation.messages) == 0 else conversation.messages[-1]
                if last_message is not None:
                    if (
                        thread == "main"
                        and float(message_obj["ts"]) - float(last_message.ts) > CONVERSATION_DISTANCE_TIME
                    ):  # Create a new conversation if enough time has passed since the previous message was sent
                        conversations.append(conversation)
                        conversation = Conversation()
                    elif message_obj["user"] == last_message.user:
                        last_message.text += ". " + text
                        continue
                conversation.messages.append(Message(text, message_obj["ts"], message_obj["user"]))
            conversations.append(conversation)
            conversation = Conversation()
    conversations = [c for c in conversations if len(c.messages) >= CONVERSATION_MIN_LENGTH]
    random.shuffle(conversations)
    train_conversations = conversations[:TRAIN_LENGTH]
    val_conversations = conversations[TRAIN_LENGTH:TRAIN_LENGTH + VAL_LENGTH]
    for file_name, data in zip([TRAIN_FILE, VAL_FILE], [train_conversations, val_conversations]):
        with open(file_name, "w", encoding="utf8") as data_file:
            json.dump(
                [{"text": "\n".join([m.text for m in d.messages])} for d in data],
                data_file, ensure_ascii=False, indent=4,
            )

generate_dataset(RAW_DATA_DIR)

import json
from datasets import load_dataset
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = 0
def load_data(file_path):
    def tokenize(d):
        encoding = tokenizer(d["text"], truncation=True, max_length=256)
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding
    data = load_dataset("json", data_files=file_path)
    tokenized_data = data["train"].shuffle().map(tokenize)
    return tokenized_data
train_data = load_data(TRAIN_FILE)
val_data = load_data(VAL_FILE)

print(len(train_data), len(val_data))

import torch
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
    ),
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
)
# model = torch.compile(model)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
