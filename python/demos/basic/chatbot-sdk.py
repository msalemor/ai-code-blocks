import os
import asyncio
from openai import AsyncAzureOpenAI
import dotenv

# Read the environment variables
dotenv.load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("GPT_MODEL")
api_version = os.getenv("API_VERSION")

client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)


async def main():
    messages = []
    while True:
        user_input = input("You (type 'exit' to break): ")
        if user_input == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        resp = response.choices[0].message.content
        messages.append({"role": "assistant", "content": resp})
        print(f"Assistant: {resp}\n\n")


if __name__ == "__main__":
    asyncio.run(main())
