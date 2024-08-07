import requests


def rest_embed(prompt: str, model='nomic-embed-text', endpoint='http://localhost:11434/api/embeddings'):
    payload = {
        "model": model,
        "prompt": prompt
    }
    req = requests.post(endpoint, json=payload)
    req.raise_for_status()
    return req.json()['embedding']


def rest_completion(prompt: str, model='phi3', endpoint='http://localhost:11434/api/chat'):
    payload = {
        "model": model,
        "messages": [
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        "stream": False
    }
    req = requests.post(endpoint, json=payload)
    req.raise_for_status()
    return req.json()['message']['content']


print(rest_embed('I am a software engineer'))
print(rest_completion('Why is the sky blue?'))
