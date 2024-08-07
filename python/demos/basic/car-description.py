import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version="2024-02-15-preview")

# Define a function to get the car details


def mock_get_car_details():
    return {"make": "Toyota", "model": "Camry", "year": 2018, "color": "blue", "price": 20000}

# Define a function to get the sales description


def get_sales_description():
    car = mock_get_car_details()

    car_str = f"A {car['color']} {car['year']} {car['make']} {car['model']} priced at ${car['price']}."

    response = client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can generate a one paragraph used car sales descriptions."},
            {"role": "user", "content": car_str}
        ],
        temperature=0.5,  # we want it somewhat creative
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    get_sales_description()
