import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# Load the environment variables
load_dotenv("../../.env")
endpoint = os.getenv("ENDPOINT")
gpt_model = os.getenv("GPT_MODEL")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION") or "2024-05-01-preview"
parser = StrOutputParser()

# Create the language model
llm = AzureChatOpenAI(
    azure_deployment=gpt_model,
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Basi chat prompt
messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French. Translate the user sentence."),
    HumanMessage(content="I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
print(parser.invoke(ai_msg))

# Chain

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

# Render the prompt template
result = prompt.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

# Show the redered messages
print(result.to_messages())

# Prepare the chain that include the prompt, the language model and the parser
# Or chain = prompt | llm without the parser
chain = prompt | llm | parser

# Invoke the chain
chain_result = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

# Print the chain response
print(chain_result)
# or print(chain_result.content) without the parser
