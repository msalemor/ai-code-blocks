import ollama


def embed(prompt: str, model='nomic-embed-text'):
    return ollama.embeddings(model=model, prompt=prompt)['embedding']


def completion(prompt: str, model='phi3'):
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return response['message']['content']


print(embed('I am a software engineer'))
print(completion('What is the speed of light?'))
