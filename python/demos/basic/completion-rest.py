import httpx
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()
full_endpoint = os.getenv("FULL_ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION") or "2024-05-01-preview"

headers = {"Content-Type": "application/json", "api-key": api_key}


async def completion(input: str, temperature: float = 0.1) -> dict:
    payload = {
        "messages": [{"role": "user", "content": input}],
        "temperature": temperature,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(full_endpoint, headers=headers, json=payload)
        return response.json()


async def main():
    response_json = await completion("What is the speed of light?")
    print(json.dumps(response_json, indent=4))
    print(response_json["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
