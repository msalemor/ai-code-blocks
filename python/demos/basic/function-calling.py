import os
import json
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Azure OpenAI configuration
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)

# Define the function schema
functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. Miami, FL",
                }
            },
            "required": ["location"],
        },
    }
]


# Simulate the function implementation
def get_weather(location):
    return {"location": location, "temperature": "88°F", "condition": "Sunny"}


async def main():
    # Chat completion with function calling
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather like in New York?"}],
        functions=functions,
        function_call="auto",
    )

    # Check if the model wants to call a function
    if response.choices[0].finish_reason == "function_call":
        function_call = response.choices[0].message.function_call
        function_name = function_call.name
        arguments = json.loads(function_call.arguments)

        # Call the function
        if function_name == "get_weather":
            result = get_weather(**arguments)

            # Send the result back to the model using the same AzureOpenAI client
            follow_up = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "What's the weather like in New York?"},
                    response.choices[0].message,
                    {
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(result),
                    },
                ],
            )

            print(follow_up.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
