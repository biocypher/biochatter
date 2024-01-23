class RagAgent:
    def __init__(self, mode):
        self.mode = mode
        if self.mode == "kg":
            from .prompts import query_language

            self.query_func = query_language
        elif self.mode == "vectorstore":
            from .vectorstore import _create_embeddings_collection

            self.query_func = _create_embeddings_collection
        else:
            raise ValueError(
                "Invalid mode. Choose either 'kg' or 'vectorstore'."
            )

    def generate_responses(self, user_question):
        return self.query_func(user_question)
