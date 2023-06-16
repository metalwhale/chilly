import glob
import os
import random
import re
import shutil
import sys
import json
import zipfile
from collections import defaultdict
from typing import List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, LlamaTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

RAW_DATA_FILE = "data/slack-export-Mart6-2018-Mar31-2023.zip"
RAW_DATA_DIR = "data/raw_data"
TRAIN_FILE = "data/train.json"
VAL_FILE = "data/val.json"
MODELS_DIR = "data/models"
LORA_RANK = 8
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
TRAIN_LENGTH = 56000
VAL_LENGTH = 1000
EPOCHS = 1

base_model = "decapoda-research/llama-7b-hf"
if len(sys.argv) >= 2:
    base_model = sys.argv[1]


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


def generate_dataset(input_dir: str) -> int:
    CONVERSATION_DISTANCE_TIME = 15 * 60  # In seconds
    CONVERSATION_MESSAGES_COUNT = 3
    TEXT_MAX_LENGTH = 384
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
        for thread, message_obj_list in thread_dict.items():
            conversation = Conversation()
            for message_obj in message_obj_list:
                if "user" not in message_obj or len(message_obj["text"]) == 0:
                    continue
                text: str = message_obj["text"]
                text = text.rstrip()
                text = re.sub(r"(\n)+", ", ", text)
                text += "."
                last_message = None if len(conversation.messages) == 0 else conversation.messages[-1]
                if last_message is not None:
                    if (
                        thread == "main"
                        and float(message_obj["ts"]) - float(last_message.ts) > CONVERSATION_DISTANCE_TIME
                    ) or len(conversation.messages) >= CONVERSATION_MESSAGES_COUNT:
                        # Create a new conversation, if enough time has passed since the previous message was sent
                        # or the max number of messages has been reached
                        conversations.append(conversation)
                        conversation = Conversation()
                    elif message_obj["user"] == last_message.user:
                        last_message.text += " " + text
                        continue
                user_name = "Chilly"
                if "user_profile" in message_obj:
                    user_name = message_obj["user_profile"]["display_name"]
                    if user_name == "":
                        user_name = message_obj["user_profile"]["real_name"]
                conversation.messages.append(Message(f"- {user_name}: {text}", message_obj["ts"], message_obj["user"]))
            conversations.append(conversation)
    texts = [
        "This is a short chat between friends in Vietnamese:\n" + "\n".join([m.text for m in c.messages])
        for c in conversations if len(c.messages) == CONVERSATION_MESSAGES_COUNT
    ]
    texts = [t for t in texts if len(t) < TEXT_MAX_LENGTH]
    random.shuffle(texts)
    train_texts = texts[:TRAIN_LENGTH]
    val_texts = texts[TRAIN_LENGTH:TRAIN_LENGTH + VAL_LENGTH]
    for file_name, data in zip([TRAIN_FILE, VAL_FILE], [train_texts, val_texts]):
        with open(file_name, "w", encoding="utf8") as data_file:
            json.dump(
                [{"text": d} for d in data],
                data_file, ensure_ascii=False, indent=4,
            )
    return len(texts)


def load_data(file_path, tokenizer):
    def tokenize(d):
        encoding = tokenizer(d["text"], truncation=True, max_length=256)
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding
    data = load_dataset("json", data_files=file_path)
    tokenized_data = data["train"].shuffle().map(tokenize)
    return tokenized_data


# Generate raw data
shutil.rmtree(MODELS_DIR, ignore_errors=True)
shutil.rmtree(RAW_DATA_DIR, ignore_errors=True)
with zipfile.ZipFile(RAW_DATA_FILE, "r") as raw_data_file:
    raw_data_file.extractall(RAW_DATA_DIR)
print("Number of conversations: ", generate_dataset(RAW_DATA_DIR))

# Create tokenizer, model, trainer and load data
tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
train_data = load_data(TRAIN_FILE, tokenizer)
val_data = load_data(VAL_FILE, tokenizer)
steps_count = TRAIN_LENGTH // BATCH_SIZE * EPOCHS
trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data if VAL_LENGTH > 0 else None,
    args=TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if VAL_LENGTH > 0 else "no",
        save_strategy="steps",
        eval_steps=steps_count // 10 if VAL_LENGTH > 0 else None,
        save_steps=steps_count // 10,
        output_dir=MODELS_DIR,
        save_total_limit=1,
        load_best_model_at_end=False,  # See: https://github.com/huggingface/peft/issues/394
    ),
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
)
model.config.use_cache = False
model = torch.compile(model)

# Train and save model
trainer.train()
model.save_pretrained(MODELS_DIR)
