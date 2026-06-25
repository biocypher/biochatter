import os
import yaml
from biochatter.prompts import BioCypherPromptEngine
from biochatter.llm_connect import LangChainConversation

with open("/Users/rupshali.dasgupta/Desktop/code/cardiometabolic-agent/biocypher-out/schema_info.yaml") as f:
    schema = yaml.safe_load(f)

connection_args = {
    "host": "localhost",
    "port": "7687",
    "db_name": "neo4j",
    "user": "",
    "password": "",
}

def conversation_factory():
    conv = LangChainConversation(
        model_provider="google_genai",
        model_name="gemini-2.5-flash",
        prompts={},
        correct=False,
    )
    conv.set_api_key(
        api_key=os.environ["GOOGLE_API_KEY"],
        user="test_user",
    )
    return conv

engine = BioCypherPromptEngine(
    schema_config_or_info_dict=schema,
    connection_args=connection_args,
    use_grounding=True,
    conversation_factory=conversation_factory,
)

question = "which genes are upregulated in AF in cardiomyocytes?"
print(f"Original question: {question}")

query = engine.generate_query(
    question=question,
    query_language="Cypher",
)

print(f"\nGenerated query:\n{query}")
