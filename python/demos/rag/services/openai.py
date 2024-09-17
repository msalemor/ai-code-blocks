from openai import AzureOpenAI
from .settings import Settings


class OpenAIHelper:
    def __init__(self, settings: Settings = None, client: AzureOpenAI = None):
        self.client = client
        self.settings = settings
        if self.settings is None:
            self.settings = Settings()
        if self.client is None:
            self.client = AzureOpenAI(
                api_key=self.settings.api_key, api_version=self.settings.api_version, azure_endpoint=self.settings.endpoint)

    def chat_completion(self, messages: list[dict] = [], model: str = "gpt-4o", temperature: float = 0.1, max_tokens: int | None = None):
        completion = self.client.chat.completions.create(
            model=model,  # e.g. gpt-35-instant
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content

    def embedding(self, input: str, model: str = 'ada') -> list[float]:
        response = self.client.embeddings.create(
            input=input,
            model=model
        )
        return response.data[0].embedding
