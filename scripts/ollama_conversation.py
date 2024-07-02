"""
Short helper script to quickly run a conversation with a self-hosted LLM via the
XinferenceConversation class.
"""

from biochatter.llm_connect import OllamaConversation

convo = OllamaConversation(
    base_url="http://localhost:11434",
    prompts={},
    correct=False,
    split_correction=False,
)

response, token_usage, correction = convo.query("Hello world!")

print(response)
print(token_usage)
