from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

def embed(prompt: str, model='nomic-embed-text'):
    text = prompt.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def completion(prompt: str, model='phi3',temperature=0.1):
    response = client.chat.completions.create(model=model, 
        messages=[
        {
            'role': 'user',
            'content': prompt,
        }],
        temperature=temperature
    )
    return response.choices[0].message.content


print(embed('I am a software engineer'))
print(completion('What is the speed of light?'))
