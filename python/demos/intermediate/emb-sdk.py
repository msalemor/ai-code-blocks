import asyncio
from openai import AsyncAzureOpenAI
import os
from dotenv import load_dotenv

# Read environment variables from .env file or the environment
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("EMB_MODEL")  # text-embedding-3-small

client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)


async def get_embedding(text, model=model) -> tuple[str, list[float]]:
    text = text.replace("\n", " ")
    response = await client.embeddings.create(input=[text], model=model)
    emb = response.data[0].embedding
    return (text, emb)


async def main():
    (text, emb) = await get_embedding(
        "App Service is on of the most used services in Azure."
    )
    print(f"Text: {text}\nEmbedding: {emb}")


if __name__ == "__main__":
    asyncio.run(main())
