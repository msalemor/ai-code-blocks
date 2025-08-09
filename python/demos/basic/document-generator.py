import os
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-15-preview"
)


async def generate_documents():
    response = await client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=[
            {
                "role": "system",
                "content": "You are a technical document writer. The user will provide a topic, and you will write a full technical document.",
            },
            {"role": "user", "content": "Prompt engineering"},
        ],
        temperature=0.1,  # we want it somewhat creative
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(generate_documents())
