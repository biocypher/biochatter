def calculate_test_score(vector: list[bool]):
    score = sum(vector)
    max = len(vector)
    return f"{score}/{max}"
