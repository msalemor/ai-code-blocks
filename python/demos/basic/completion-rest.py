import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
# Full endpoint format:
#   https://<NAME>.openai.azure.com/openai/deployments/<MODEL>/chat/completions?api-version=2024-02-15-preview
# os.getenv("FULL_ENDPOINT")
full_endpoint = os.getenv("FULL_ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION") or "2024-05-01-preview"

# Set the headers and authentication
headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}


def completion(input: str, temperature: float = 0.1) -> dict:
    # Construct the request payload
    payload = {
        "messages": [
            {
                "role": "user",
                "content": input
            }
        ],
        "temperature": temperature
    }
    # Make the REST call
    response = requests.post(full_endpoint, headers=headers, json=payload)
    # Get the response JSON
    return response.json()


# Set the prompt and other parameters
response_json = completion("What is the speed of light?")

# Print the full JSON response
print(json.dumps(response_json, indent=4))

# Print the response only
print(response_json['choices'][0]['message']['content'])
