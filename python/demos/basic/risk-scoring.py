import os
import json
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)


def get_mock_document() -> str:
    return """
    Incident Summary:
    On June 12, 2024, at approximately 09:15 UTC, monitoring systems observed an unusual increase in inbound network traffic targeting public-facing web services. Telemetry data showed a higher than normal volume of requests from a diverse set of IP addresses. The traffic pattern is atypical. Investigation is ongoing to determine the nature and intent of the observed behavior, and precautionary monitoring measures have been implemented.
"""


async def evaluate(content: str) -> float:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'You are a risk assessment evaluator. The user will provide a summary of the condition(s) and you need to evaluate the risk level. Provide a score from 0 to 1 with 1 indicating a risky condition.\nNo prologue. Respond in the following JSON format:\n{"score":"","reason":"" }.',
            },
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(asyncio.run(evaluate(get_mock_document())))
