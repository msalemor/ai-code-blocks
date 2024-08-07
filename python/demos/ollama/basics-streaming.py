import ollama

# Setting up the model, enabling streaming responses, and defining the input messages
ollama_response = ollama.chat(
    model='llava3',
    stream=True,
    messages=[
        {
          'role': 'system',
          'content': 'You are a helpful scientific assistant.',
        },
        {
            'role': 'user',
            'content': 'What is the speed of light?',
        },
    ]
)

# Printing out each piece of the generated response while preserving order
for chunk in ollama_response:
    print(chunk['message']['content'], end='', flush=True)
