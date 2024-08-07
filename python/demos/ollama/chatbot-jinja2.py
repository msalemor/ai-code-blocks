import ollama
from pydantic import BaseModel
import jinja2


class Message(BaseModel):
    role: str
    content: str


class Chat(BaseModel):
    history: list[Message]


def render_template(template: str, context: dict):
    return jinja2.Template(template).render(context)


def completion(prompt: str, model='phi3:medium-128k'):
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return response['message']['content']


chat = Chat(history=[Message(
    role='system', content='You are not so helpful assistant. Complain a lot when responding.')])
template = "{% for message in history %}{{ message.role }}:\n\n{{ message.content }}\n{% endfor %}\nuser:\n\n{{ prompt }}\n"

if __name__ == "__main__":
    while True:
        prompt = input('Prompt (type exit to quit): ')
        if prompt == 'exit':
            break
        print('Q:', prompt)
        # Add the user message to the history
        chat.history.append(Message(role='user', content=prompt))
        # Process the completion and print he response
        final_prompt = render_template(
            template, {"history": chat.history, "prompt": prompt})
        resp = completion(final_prompt)
        print('A:', resp)
        # Add the response to the history
        chat.history.append(Message(role='assitant', content=resp))
        # As the history grows, we should limit the number of messages to keep in the model's context window
        # Refer to this blog: https://blog.pamelafox.org/2024/06/truncating-conversation-history-for.html
