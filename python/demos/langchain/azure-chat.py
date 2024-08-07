import os

from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# Load the environment variables
load_dotenv("../../.env")
endpoint = os.getenv("ENDPOINT")
gpt_model = os.getenv("GPT_MODEL")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION") or "2024-05-01-preview"
parser = StrOutputParser()


# Create the language model
model = AzureChatOpenAI(
    azure_deployment=gpt_model,
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Manual handling
response = model.invoke(
    [
        SystemMessage(
            content="You are a travel assistant that help users with travel destination information."),
        HumanMessage(content="What are some restaurants in Miami?"),
        AIMessage(content="1. **Zuma** - Contemporary Japanese cuisine located in the heart of downtown Miami.\n2. **KYU** - Known for its wood-fired Asian-inspired dishes in the Wynwood Arts District."),
        HumanMessage(content="Ware are some more?"),
    ]
)
print(response.content)


# Using RunnableWithMessageHistory
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "abc2"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)

print(response.content)

response = with_message_history.invoke(
    [HumanMessage(content="What is my name")],
    config=config,
)

print(response.content)

# Using RunnableWithMessageHistory with a prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="hi! I'm todd")],
     "language": "Spanish"},
    config=config,
)

print(response.content)


# Using RunnablePassthrough and a trimmer
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)


messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    # AIMessage(content="nice"),
    # HumanMessage(content="whats 2 + 2"),
    # AIMessage(content="4"),
    # HumanMessage(content="thanks"),
    # AIMessage(content="no problem!"),
    # HumanMessage(content="having fun?"),
    # AIMessage(content="yes!"),
]

response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)
print(response.content)
