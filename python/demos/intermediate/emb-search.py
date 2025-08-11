import math
import asyncio
from openai import AsyncAzureOpenAI
import os
from dotenv import load_dotenv

# Read environment variables from .env file or the environment
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("EMB_MODEL")  # text-embedding-3-small

client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)


async def get_embedding(text, model=model) -> tuple[str, list[float]]:
    text = text.replace("\n", " ")
    response = await client.embeddings.create(input=[text], model=model)
    emb = response.data[0].embedding
    return (text, emb)


def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    # Calculate the dot product of the two embeddings
    dot_product = sum(x * y for x, y in zip(embedding1, embedding2))

    # Calculate the magnitudes of the two embeddings
    magnitude1 = math.sqrt(sum(x**2 for x in embedding1))
    magnitude2 = math.sqrt(sum(x**2 for x in embedding2))

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


async def main():
    content = [
        "Azure App Service enables you to host web applications in the cloud.",
        "Azure Functions allows you to run event-driven serverless code.",
        "Azure Logic Apps helps automate workflows and integrate services.",
        "Azure SQL Database is a fully managed relational database service.",
    ]

    ram_vector_database = [await get_embedding(c) for c in content]

    question = "What a PaaS database service in Azure?"
    (query_content, embedding) = await get_embedding(question)

    # Perform near search with relevance and limits
    limit = 3
    relevance = 0.5
    results_list = []
    for entry in ram_vector_database:
        (content, entry_embedding) = entry
        cs = cosine_similarity(embedding, entry_embedding)
        if cs >= relevance:
            results_list.append((content, cs))

    # print the results
    results_list.sort(key=lambda x: x[1], reverse=True)
    top_n = results_list[:limit]
    (result, score) = top_n[0]
    print(f"Question: {question}\nTop result: {result}\nScore: {score}")


if __name__ == "__main__":
    asyncio.run(main())
