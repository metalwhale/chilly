from collections import defaultdict
import glob
import json
import os
import random
import sys


CONVERSATION_DISTANCE_TIME = 30  # in seconds
CONVERSATION_MIN_LENGTH = 2


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

    def append_message(self, message: Message):
        self.messages.append(message)

    @property
    def last_message(self) -> Message:
        return None if len(self.messages) == 0 else self.messages[-1]


class Instruction:
    instruction: str
    input: str
    output: str

    def __init__(self, instruction: str, output: str) -> "Instruction":
        self.instruction = instruction
        self.input = ""
        self.output = output


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
                if conversation.last_message is not None:
                    if (
                        thread == "main"
                        and float(message_obj["ts"]) - float(conversation.last_message.ts) > CONVERSATION_DISTANCE_TIME
                    ):  # Create a new conversation if enough time has passed since the previous message was sent
                        conversations.append(conversation)
                        conversation = Conversation()
                    elif message_obj["user"] == conversation.last_message.user:
                        conversation.last_message.text += "\n" + message_obj["text"]
                        continue
                conversation.append_message(Message(message_obj["text"], message_obj["ts"], message_obj["user"]))
            conversations.append(conversation)
            conversation = Conversation()
    return conversations


def convert_to_instruction_list(conversations: list[Conversation]) -> list[Instruction]:
    instructions: list[Instruction] = []
    for conversation in conversations:
        messages = conversation.messages
        if len(messages) < CONVERSATION_MIN_LENGTH:
            continue
        instructions.append(Instruction("\n".join([m.text for m in messages[:-1]]), messages[-1].text))
    random.shuffle(instructions)
    return instructions


if __name__ == "__main__":
    conversations = load_workspace_data(sys.argv[1])
    instructions = convert_to_instruction_list(conversations)
    print("Instructions size:", len(instructions))
    with open(sys.argv[2], "w", encoding="utf8") as instructions_file:
        json.dump([instruction.__dict__ for instruction in instructions], instructions_file, ensure_ascii=False, indent=4)
