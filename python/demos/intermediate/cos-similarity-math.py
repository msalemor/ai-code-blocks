import math


def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    # Calculate the dot product of the two embeddings
    dot_product = sum(x * y for x, y in zip(embedding1, embedding2))

    # Calculate the magnitudes of the two embeddings
    magnitude1 = math.sqrt(sum(x**2 for x in embedding1))
    magnitude2 = math.sqrt(sum(x**2 for x in embedding2))

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


# Example usage:
embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
embedding2 = [0.6, 0.7, 0.8, 0.9, 1.0]

similarity = cosine_similarity(embedding1, embedding2)
print(similarity)

embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
embedding2 = [0.1, 0.2, 0.3, 0.4, 0.5]

similarity = cosine_similarity(embedding1, embedding2)
print(similarity)
