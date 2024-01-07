import glob
import json
import os
import random
import re
from collections import defaultdict
from typing import List


TRAIN_FILE = "data/train.json"
VAL_FILE = "data/val.json"


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


def generate_dataset(
    input_dir: str, train_len: int, val_len: int,
    conversation_time_gap: int = 15 * 60,  # In seconds
    conversation_messages_len: int = 2,
    text_max_len: int = 384,
) -> int:
    # Divide the chat history into conversations
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
                    # Normal messages. See: https://api.slack.com/events/message#subtypes.
                    elif "subtype" not in message_obj:
                        thread_dict["main"].append(message_obj)
        for thread, message_obj_list in thread_dict.items():
            conversation = Conversation()
            for message_obj in message_obj_list:
                if "user" not in message_obj or len(message_obj["text"]) == 0:
                    continue
                text = _preprocess(message_obj["text"])
                last_message = None if len(conversation.messages) == 0 else conversation.messages[-1]
                if last_message is not None:
                    if (
                        thread == "main"
                        and float(message_obj["ts"]) - float(last_message.ts) > conversation_time_gap
                    ) or len(conversation.messages) >= conversation_messages_len:
                        # Create a new conversation, if enough time has passed since the previous message was sent
                        # or the max number of messages has been reached
                        conversations.append(conversation)
                        conversation = Conversation()
                    elif message_obj["user"] == last_message.user:
                        last_message.text += " " + text
                        continue
                conversation.messages.append(Message(text, message_obj["ts"], message_obj["user"]))
            conversations.append(conversation)
    # Convert to strings
    texts = [
        "\n".join([m.text for m in c.messages])
        for c in conversations if len(c.messages) == conversation_messages_len
    ]
    texts = [t for t in texts if len(t) < text_max_len]
    random.shuffle(texts)
    # Split into train and val
    train_texts = texts[:train_len]
    val_texts = texts[train_len:train_len + val_len]
    for file_name, data in zip([TRAIN_FILE, VAL_FILE], [train_texts, val_texts]):
        with open(file_name, "w", encoding="utf8") as data_file:
            json.dump([{"text": d} for d in data], data_file, ensure_ascii=False, indent=4)
    return len(texts)


def _preprocess(text: str) -> str:
    text = text.rstrip()
    text = re.sub(r"(\n)+", ", ", text)
    text += "."
    return text
