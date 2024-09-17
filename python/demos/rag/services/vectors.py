import math


def similarity(vector1: list[float], vector2: list[float]) -> float:
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vector1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vector2))
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity
