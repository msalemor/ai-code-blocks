from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI
import os
import openai

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)

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
def post_completion(request: PromptRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=404, detail="Messages required")
    response = client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=request.messages,
        temperature=request.temperature,  # less creative
    )
    # Print and add the response to the messages
    resp = response.choices[0].message.content
    return CompletionResponse(response=resp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
