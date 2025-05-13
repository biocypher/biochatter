"""Short helper script to quickly run a conversation with a self-hosted LLM via the
XinferenceConversation class.
"""

from biochatter.llm_connect import XinferenceConversation

convo = XinferenceConversation(
    base_url="http://localhost:9997",
    prompts={},
    correct=False,
    split_correction=False,
)

query_result = convo.query("Hello world!")

print(query_result.response)
print(query_result.token_usage)
