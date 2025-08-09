from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import os

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class PromptRequest(BaseModel):
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float = 0.1


class CompletionResponse(BaseModel):
    response: str


@app.post("/completion", response_model=CompletionResponse)
async def post_completion(request: PromptRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=404, detail="Messages required")
    response = await client.chat.completions.create(
        model=model,
        messages=request.messages,
        temperature=request.temperature,
    )
    resp = response.choices[0].message.content
    return CompletionResponse(response=resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
