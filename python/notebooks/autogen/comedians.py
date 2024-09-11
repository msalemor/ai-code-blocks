import os

from autogen import ConversableAgent
from settings import Settings
settings = Settings()

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

cathy = ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians.",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
)

joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config=llm_config,
    #max_consecutive_auto_reply=2,
    human_input_mode="NEVER",  # Never ask for human input.
)

#result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.", max_turns=2)
result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.")