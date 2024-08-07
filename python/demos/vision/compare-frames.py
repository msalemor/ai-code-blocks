import base64
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv("../../.env")

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("GPT_MODEL")
api_version = os.getenv("API_VERSION") or "2024-05-01-preview"

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)


def get_completion(prompt: str, base64_image: str | None = None, temperature=0.1):
    if base64_image is None:
        # Create the completion
        response = client.chat.completions.create(
            model=model,
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
            model=model,
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


def describe_frame(base64_image_content: str):
    prompt = "Describe the contents of the image."
    return get_completion(prompt, base64_image_content)


def file_to_base64(file_path: str):
    # Read the file
    with open(file_path, 'rb') as file:
        # Convert the file to base64
        return base64.b64encode(file.read()).decode('utf-8')
    return None


def diff_frames_descs(desc1: str, desc2: str) -> str:
    prompt = """system:
You are an agent that compares two descriptions that came from analyzing video frames. Point out the differences between the two descriptions particularly around what a person or people may be doing. Highligh the differences in the actions of the people in the two frames. 

Danger condition rules:
- A person laying on the floor or sitting down
- A person giving a thumbs down
- A person or people running
- Any other danger condition is detected.

Output in JSON the following JSON format:

{
    "summary":"",
    "danger_conditions":[]
}

user:

Description 1: \"\"\"
<desc1>
\"\"\"

Description 2: \"\"\"
<desc2>
\"\"\"
"""
    prompt = prompt.replace("<desc1>", desc1).replace("<desc2>", desc2)
    return get_completion(prompt)


if __name__ == '__main__':
    target_folder = "."
    # Print the base64 string
    base64_frame1 = file_to_base64(
        target_folder + '/frames/frame1.jpg')
    base64_frame2 = file_to_base64(
        target_folder+'/frames/frame2.jpg')

    desc1 = describe_frame(base64_frame1)
    desc2 = describe_frame(base64_frame2)

    diff = diff_frames_descs(desc1, desc2)
    print(diff)
