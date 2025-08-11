import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    azure_endpoint=endpoint, api_key=api_key, api_version=api_version
)


# Make a completion request
async def completion(input: str, temperature: float = 0.1) -> tuple[dict, str]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": input,
            },
        ],
    )
    return (json.loads(completion.to_json()), completion.choices[0].message.content)


# Set the prompt and other parameters
async def main():
    full, response = await completion("What are some Azure compute services?")
    print(json.dumps(full, indent=4))
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
