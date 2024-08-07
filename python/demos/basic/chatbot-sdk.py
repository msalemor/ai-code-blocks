import os
from openai import AzureOpenAI
import dotenv

# Read the enviroment variables
dotenv.load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("GPT_MODEL")
api_version = os.getenv("API_VERSION")

client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)

if __name__ == "__main__":
    messages = []
    while True:
        # Get the user input
        user_input = input("You (type 'exit' to break): ")
        if user_input == "exit":
            break
        # Add the user input to the messages
        messages.append({"role": "user", "content": user_input})
        # Call GPT with the messages
        response = client.chat.completions.create(
            model=model,  # model = "deployment_name".
            messages=messages,
            temperature=0.3,  # less creative
        )
        # Print and add the response to the messages
        resp = response.choices[0].message.content
        messages.append({"role": "assistant", "content": resp})
        print(f"Assistant: {resp}\n\n")
