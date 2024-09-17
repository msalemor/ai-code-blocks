import os
from services.openai import *
from services.vectors import similarity

executing_file_path = os.path.abspath(__file__)
faq = os.path.join(os.path.dirname(
    executing_file_path), "data", "faq.txt")
print(f"Executing data path: {faq}")


def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()


def split_paragraphs(text: str) -> list[str]:
    return text.split("\n\n")


def create_vectordb(paragraphs: list[str], helper: OpenAIHelper) -> list[list[float]]:
    db = []
    for paragraph in paragraphs:
        item = {"chunk": paragraph, "vector": helper.embedding(
            paragraph), "file": faq}
        db.append(item)
    return db


def query(question: str, vectordb: list[dict], helper: OpenAIHelper, limit: int = 2, relevance: float = 0.7) -> str:
    # Embed the question
    question_vector = helper.embedding(question)
    results = []
    # Calculate similarity between the chunks and the question
    for item in vectordb:
        similarity_score = similarity(question_vector, item["vector"])
        item["relevance"] = similarity_score
        results.append(item)
    results.sort(key=lambda x: x["relevance"], reverse=True)
    # Filter by relevance
    results = [item for item in results if item["relevance"] > relevance]
    # Limit results
    results = results[:limit]
    return results


if __name__ == "__main__":

    helper = OpenAIHelper()

    # Ingestion
    contents = read_file(faq)
    paragraphs = split_paragraphs(contents)
    vectordb = create_vectordb(paragraphs, helper)  # poor man vector database

    # Question
    question = "What is the return policy?"
    question_vector = helper.embedding(question)

    # Query
    results = query(question, vectordb, helper)

    # Prompt augmentation
    prompt = question+"\\n\\n"+results[0]["chunk"]

    # Completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can answer FAQ question based on the provided content. If you do not know the answer, you can say 'I do not know'."},
        {"role": "user", "content": prompt},
    ]
    print(f"Question: {question}")
    print(f"Answer: {helper.chat_completion(messages=messages)}")

    print("Done.")
