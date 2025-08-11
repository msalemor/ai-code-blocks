import os
import asyncio
import httpx
from dotenv import load_dotenv

# Read environment variables from .env file or the environment
load_dotenv()
emb_full_endpoint = os.getenv("EMB_FULL_ENDPOINT")
api_key = os.getenv("API_KEY")

headers = {"Content-Type": "application/json", "api-key": api_key}


async def get_embedding(
    input: str, model_version=2, dimensions=1536
) -> tuple[str, list[float]]:
    json_data = {"input": input}
    if model_version == 3:
        json_data = {"input": input, "dimensions": dimensions}

    async with httpx.AsyncClient() as client:
        response = await client.post(emb_full_endpoint, headers=headers, json=json_data)
        response.raise_for_status()
        res = response.json()

    vector = res["data"][0]["embedding"]
    return (input, vector)


async def main():
    (text, emb) = await get_embedding(
        "App Service is on of the most used services in Azure."
    )
    print(f"Input: {text} Vector\n: {emb}")


if __name__ == "__main__":
    asyncio.run(main())
