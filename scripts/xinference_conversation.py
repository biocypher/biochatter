from biochatter.llm_connect import XinferenceConversation

convo = XinferenceConversation(
    base_url="https://llm.biocypher.org",
    prompts={},
    correct=False,
    split_correction=False,
)

convo.set_api_key("none")
response, token_usage, correction = convo.query("Hello world!")

print(response)
print(token_usage)
