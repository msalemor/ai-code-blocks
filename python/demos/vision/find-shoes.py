import base64
import math
import os
from time import sleep
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

# Load the environment variables
load_dotenv("../../.env")

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
gpt_model = os.getenv("GPT_MODEL")
ada_model = os.getenv("ADA_MODEL")
api_version = os.getenv("API_VERSION") or "2024-05-01-preview"
root_folder = "."

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)

def get_embedding(text:str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=ada_model).data[0].embedding

def get_completion(prompt: str, base64_image: str | None = None, temperature=0.1):
    if base64_image is None:
        # Create the completion
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt}
                 ]}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt},
                     {"type": "image_url", "image_url": {
                         "url": f"data:image/png;base64,{base64_image}"}
                      }
                 ]}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

def decribe_image(base64_image_content: str, 
                  prompt:str = "Describe the contents of the image.") -> str:
    return get_completion(prompt, base64_image_content)


def get_file_base64(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        # Convert the file to base64
        return base64.b64encode(file.read()).decode('utf-8')

def generate_shoe_memories(folder:str) -> list:
    # List files in the folder
    files = os.listdir(folder)
    shoes = []
    for file in files:
        # Read the file
        filePath = folder+ "/"+ file
        # Convert the file to base64
        base64_image = get_file_base64(filePath)
        # Describe the frame
        description = decribe_image(base64_image,"The following is an image of a shoe. Describe the shoe including the color and the type. Ignore everything else.")
        # Get the embedding
        embedding = get_embedding(description)
        # Store the embedding
        shoes.append({"description": description, "embedding": embedding, "file": filePath})
    return shoes

def cosine_similarity(vector1: list[float], vector2: list[float])->float:
    dot_product = 0
    norm1 = 0
    norm2 = 0

    for i in range(len(vector1)):
        dot_product += vector1[i] * vector2[i]
        norm1 += vector1[i] ** 2
        norm2 += vector2[i] ** 2

    similarity = dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    return similarity

def get_results(input_embedding: list[float], vectordb: list, relevance=0.75)->list:
    results = []
    relevance = relevance
    for shoe in vectordb:
        similarity = cosine_similarity(input_embedding, shoe["embedding"])
        if similarity>relevance:
            results.append({"file": shoe["file"], "description": shoe["description"], "relevance": similarity})
    # order by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results

def print_results(results):
    for shoe in results:
        print(shoe["relevance"],shoe["file"]," ",shoe["description"])
        
def serilize(items: list[dict], file_path: str = 'cache.data'):
    if len(items)==0:
        return
    with open(root_folder+"/"+file_path, 'w') as file:
        json.dump(items, file)
                  
def deserialize(file_path: str = 'cache.data') -> list[dict]:
    try:
        with open(root_folder+"/"+file_path, 'r') as file:
            data = json.load(file)
            if data is None:
                return []
            return data
    except:
        return []

def mock_take_photo():
    return root_folder+"/"+"input_image.png"

if __name__=="__main__":

    cache = deserialize()
    if len(cache)>0: # Cache hit
        poorman_vectordb = cache
    else: # Cache miss
        poorman_vectordb = generate_shoe_memories(root_folder+"/shoes")
        serilize(poorman_vectordb)

    # Find a shoe based on a user input
    user_input = "Can you recommend red pumps?"
    user_input_embedding = get_embedding(user_input)

    # Get and print the results
    print("User input: ", user_input)
    results= get_results(user_input_embedding, poorman_vectordb)    
    print_results(results)

    # Find a shoe based on a user taking a photo    
    input_image = mock_take_photo()
    input_image_base64 = get_file_base64(input_image)
    image_description = decribe_image(input_image_base64, "The following is an image of a product. Describe the the product. Ignore everything else.")
    user_input_embedding = get_embedding("Find a shoe that looks like the following description: "+image_description)
    
    # Get and print the results
    print("Picture taken input: ", image_description)
    results= get_results(user_input_embedding,poorman_vectordb)
    print_results(results)
