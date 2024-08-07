import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)


# Define a function to get the product comments
def mock_get_product_comments() -> list[str]:
    return ["I love this product!",
            "This product is terrible.",
            "This product is okay."]


def get_sentiment_score(comment: str) -> float:
    response = client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can perform sentiment analysis. Analyze the sentiment and provide a score from 0 to 10 with 10 being best.\nNo prologue. Respond in the following JSON format:\n{\"score\": }."},
            {"role": "user", "content": comment}
        ],
        temperature=0.1,
    )
    json_respond = response.choices[0].message.content
    print(json_respond)
    sentiment = json.loads(json_respond)
    return float(sentiment["score"])


def get_sentiment():
    comments = mock_get_product_comments()
    total = 0.0
    for comment in comments:
        total += get_sentiment_score(comment)
    print(f"Average sentiment score: {total / len(comments)}")


if __name__ == "__main__":
    get_sentiment()
