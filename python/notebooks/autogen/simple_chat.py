from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import autogen
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import ConversableAgent, ChatResult
from settings import Settings
settings = Settings()


chat_history = []


def process_message(
    sender: autogen.Agent,
    receiver: autogen.Agent,
    message: Dict,
    request_reply: bool = False,
    silent: bool = False,
    sender_type: str = "agent",
) -> None:
    """
    Processes the message and adds it to the agent history.

    Args:

        sender: The sender of the message.
        receiver: The receiver of the message.
        message: The message content.
        request_reply: If set to True, the message will be added to agent history.
        silent: determining verbosity.
        sender_type: The type of the sender of the message.
    """

    message = message if isinstance(message, dict) else {
        "content": message, "role": "user"}
    message_payload = {
        "recipient": receiver.name,
        "sender": sender.name,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "sender_type": sender_type,
        "message_type": "agent_message",
    }
    chat_history.append(message_payload)


class ExtendedConversableAgent(ConversableAgent):
    def __init__(self, message_processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_processor = message_processor

    def receive(
        self,
        message: Union[Dict, str],
        sender: autogen.Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        if self.message_processor:
            self.message_processor(sender, self, message,
                                   request_reply, silent, sender_type="agent")
        # print(f"Sender: {sender.name}")
        # print(f"Message: {message}")
        super().receive(message, sender, request_reply, silent)


config_list = [
    {
        "base_url": settings.endpoint,
        "api_key": settings.api_key,
        "model": settings.model,
        "api_type": "azure",
        "api_version": settings.api_version
    }
]

llm_config = {
    "model": settings.model,
    "temperature": 0,
    "config_list": config_list,
    "cache_seed": None,
}

# Local code execution
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)
executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

# Create the agent that uses the LLM.
assistant = ExtendedConversableAgent(
    name="agent",
    # code_execution_config={"executor": executor},
    llm_config=llm_config,
    message_processor=process_message,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)

# Create the agent that represents the user in the conversation.
user = ExtendedConversableAgent(name="user",
                                max_consecutive_auto_reply=1,
                                # code_execution_config=False,
                                code_execution_config={"executor": executor},
                                # is_termination_msg=lambda msg: msg.get(
                                #     "content") is not None and "TERMINATE" in msg["content"],
                                human_input_mode="NEVER",
                                message_processor=process_message,
                                )


def process(message: str, silent=True, clear_history=False, max_turns=1):
    # Load LLM inference endpoints from an env variable or a file
    # See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
    # and OAI_CONFIG_LIST_sample.
    # For example, if you have created a OAI_CONFIG_LIST file in the current working directory, that file will be used.
    # config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

    # Let the assistant start the conversation.  It will end when the user types exit.
    print("Q:", message)
    res: ChatResult = user.initiate_chat(
        assistant,
        message=message,
        silent=silent,
        # summary_method="reflection_with_llm",
        clear_history=clear_history)

    # print(res)
    # return res.chat_history
    # return res.chat_history[-1]['content']
    return res.summary

# Workflow
# user(Message) -> <- Assistant(Reply)
#   No LLM but with code executor
#   Sends messages to Assistant
#   Assistant
#     LLM and no tools
#     Sends messages back to User


def find_all_messages(messages: list) -> str:
    l = len(messages)
    if l == 0:
        return ""

    list = []
    for i in range(l-1, -1, -1):
        role = messages[i]['role']
        if role == 'assistant':
            break
        if role == 'user':
            list.append(messages[i]['content'])
    list.reverse()
    return list.join("\n")


def print_chat_history():
    for entry in chat_history:
        message = entry['message']
        content = message['content']
        sender = entry['sender']
        if content != "":
            print(f"Sender: {sender}\nContent:\n{content}\n")


if __name__ == "__main__":
    process("List one top seafood restaurant in Miami.")
    process("List another restaurant.")
    process("Using Python count from 1 to 10.")
    print_chat_history()
    # print(user.chat_messages)
    # print("Assistant:\n", assistant.chat_messages[user][-1])
    # print("User:\n", user.chat_messages[assistant][-1])
