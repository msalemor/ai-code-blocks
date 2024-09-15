from openai import AzureOpenAI

client = AzureOpenAI()

def completion(messages: list[dict]=[],temperature:float=0.1,max_tokens:int | None=None):
    response = client.completions.create(
        model='',
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content