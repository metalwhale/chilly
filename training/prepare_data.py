from collections import defaultdict
import glob
import json
import os
import random
import re
import sys


CONVERSATION_DISTANCE_TIME = 15 * 60  # in seconds
CONVERSATION_MIN_LENGTH = 2
TRAIN_VAL_RATIO = (9, 1)


class Message:
    text: str
    ts: str  # timestamp
    user: str

    def __init__(self, text: str, ts: str, user: str) -> "Message":
        self.text = text
        self.ts = ts
        self.user = user


class Conversation:
    messages: list[Message]

    def __init__(self) -> "Conversation":
        self.messages = []


def load_workspace_data(input_folder: str) -> list[Conversation]:
    """
    Parameters:
        input_folder: extracted from the .zip file that contains workspace data
    """
    conversations: list[Conversation] = []
    for channel in next(os.walk(input_folder))[1]:  # List all subdirectories
        thread_dict = defaultdict(list)
        for json_file_path in glob.glob(os.path.join(input_folder, channel, "*.json")):
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
    return conversations


if __name__ == "__main__":
    conversations = load_workspace_data(sys.argv[1])
    random.shuffle(conversations)
    train_length = int(TRAIN_VAL_RATIO[0] / sum(TRAIN_VAL_RATIO) * len(conversations))
    train_conversations = conversations[:train_length]
    val_conversations = conversations[train_length:]
    print("Train length:", len(train_conversations))
    print("Val length:", len(val_conversations))
    for name, data in zip(["train", "val"], [train_conversations, val_conversations]):
        with open(os.path.join(sys.argv[2], f"{name}.json"), "w", encoding="utf8") as data_file:
            json.dump(["\n".join([m.text for m in d.messages]) for d in data], data_file, ensure_ascii=False, indent=4)
