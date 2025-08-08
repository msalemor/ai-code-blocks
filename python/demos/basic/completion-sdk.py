import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

# Make a completion request


def completion(input: str, temperature: float = 0.1) -> tuple[dict, str]:
    completion = client.chat.completions.create(
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
(full, response) = completion("What is the speed of light?")

# Print the full JSON response
print(json.dumps(full, indent=4))

# Print the response only
print(response)
