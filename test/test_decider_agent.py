
import json
from typing import Dict, List
from langchain_openai import AzureOpenAIEmbeddings
import pytest
from unittest.mock import patch, MagicMock
import os
from dotenv import load_dotenv
import neo4j_utils as nu
import logging
from langchain.schema import Document
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
)
from langchain.output_parsers.openai_tools import (
    PydanticToolsParser,
)
import shortuuid

from biochatter.decider_agent import ChooseRagAgent, DeciderAgent, ReviseRagAgent
from biochatter.langgraph_agent_base import ReflexionAgentResult, ResponderWithRetries
from biochatter.llm_connect import AzureGptConversation
from biochatter.rag_agent import RagAgent, RagAgentModeEnum

load_dotenv()

logger = logging.getLogger(__name__)

def find_schema_info_node(connection_args: dict):
    try:
        """
        Look for a schema info node in the connected BioCypher graph and load the
        schema info if present.
        """
        db_uri = "bolt://" + connection_args.get("host") + \
            ":" + connection_args.get("port")
        neodriver = nu.Driver(
            db_name=connection_args.get("db_name") or "neo4j",
            db_uri=db_uri,
        )
        result = neodriver.query("MATCH (n:Schema_info) RETURN n LIMIT 1")

        if result[0]:
            schema_info_node = result[0][0]["n"]
            schema_dict = json.loads(schema_info_node["schema_info"])
            return schema_dict

        return None
    except Exception as e:
        logger.error(e)
        return None

def create_conversation():
    chatter = AzureGptConversation(
        deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"],
        model_name=os.environ["OPENAI_MODEL"],
        prompts={"rag_agent_prompts": ""},
        version=os.environ["OPENAI_API_VERSION"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    chatter.set_api_key(os.environ["OPENAI_API_KEY"])
    return chatter


@pytest.fixture
def vectorAgent():
    return RagAgent(
        mode=RagAgentModeEnum.VectorStore,
        connection_args={
            "host": "10.95.224.94",
            "port": "19530",
        },
        conversation_factory=create_conversation,
        embedding_func=AzureOpenAIEmbeddings(
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
            model=os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL"]
        ),
        model_name="gpt-3.5-turbo",
        use_prompt=True,
    )

@pytest.fixture
def apiBlastAgent():
    return RagAgent(
        mode=RagAgentModeEnum.API_BLAST,
        conversation_factory=create_conversation,
        use_prompt=True,
    )
@pytest.fixture
def apiOncoKBAgent():
    return RagAgent(
        mode=RagAgentModeEnum.API_ONCOKB,
        conversation_factory=create_conversation,
        use_prompt=True,
    )

@pytest.fixture
def databaseAgent():
    connection_args={
        "host": "10.95.224.94",
        "port": "47687",
    }
    schema_dict = find_schema_info_node(connection_args)
    return RagAgent(
        mode=RagAgentModeEnum.KG,
        connection_args=connection_args,
        schema_config_or_info_dict=schema_dict,
        model_name="gpt-3.5-turbo",
        conversation_factory=create_conversation,
        use_prompt=True,
    )
@pytest.mark.skip()
def test_decider_agent_kg(vectorAgent, databaseAgent):
    decider_agent = DeciderAgent(
        rag_agents=[vectorAgent, databaseAgent],
        conversation_factory=create_conversation
    )
    result = decider_agent.execute("what are the biological functions of the gene SETBP1")
    assert result is not None
    logs = decider_agent.get_logs()
    with open("./temp-3.log", "a+") as fobj:
        fobj.write(logs)

@pytest.mark.skip()
def test_decider_agent_vectorstore(vectorAgent, databaseAgent):
    decider_agent = DeciderAgent(
        rag_agents=[vectorAgent, databaseAgent],
        conversation_factory=create_conversation
    )
    result = decider_agent.execute("How to reset trip for Lexus LS 460 after maintenance")
    assert result is not None
    logs = decider_agent.get_logs()
    with open("./temp-3.log", "a+") as fobj:
        fobj.write(logs)

@pytest.mark.skip()
def test_decider_agent_api(
        vectorAgent, 
        databaseAgent, 
        apiBlastAgent, 
        apiOncoKBAgent
    ):
    decider_agent = DeciderAgent(
        rag_agents=[
            vectorAgent, databaseAgent, apiBlastAgent, apiOncoKBAgent
        ],
        conversation_factory=create_conversation
    )
    result = decider_agent.execute("What is the oncogenic potential of BRAF V600E mutation?")
    assert result is not None
    logs = decider_agent.get_logs()
    with open("./temp-3.log", "a+") as fobj:
        fobj.write(logs)

class ChatOpenAIMock:
    def __init__(self) -> None:
        self.chat = None

class InitialResponder:
    def __init__(self, rag_agent: str):
        self.rag_agent = rag_agent
    def invoke(self, msg_obj: Dict[str, List[BaseMessage]]) -> BaseMessage:
        msg = AIMessage(
            content="initial test"
        )
        id = 'call_' + shortuuid.uuid()
        msg.additional_kwargs = {'tool_calls': [{
            'id': id, 'function': {'arguments': 
                                   '{"answer":"' + f'{self.rag_agent}' + '","reflection":"balahbalah"}', 'name': 'ChooseRagAgent'}, 
            'type': 'function'}]}
        msg.id = id
        msg.tool_calls = [{
        'name': 'ChooseRagAgent', 
        'args': {'answer': f'{self.rag_agent}', 'reflection': "balahbalah"}, 
        'id': id
        }]
        return msg
    
class ReviseResponder:
    def __init__(self, rag_agent: str):
        self.rag_agent = rag_agent
    def invoke(self, msg_obj: Dict[str, List[BaseMessage]]) -> BaseMessage:
        return AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_wTO40b9', 'function': {'arguments': '{"answer":"' + f'{self.rag_agent}' + '","reflection":"balahbalah.","revised_answer":"' + f'{self.rag_agent}' + '","score":"10","tool_result":"balahbalah"}', 'name': 'ReviseRagAgent'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 146, 'prompt_tokens': 419, 'total_tokens': 565}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_abc28019ad', 'prompt_filter_results': [{'prompt_index': 0, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}], 'finish_reason': 'stop', 'logprobs': None, 'content_filter_results': {}}, id='run-ad95b309-4f99-4886-b70f-bcb8b59f537f-0', tool_calls=[{'name': 'ReviseRagAgent', 'args': {'answer': f'{self.rag_agent}', 'reflection': 'balahbalah', 'revised_answer': f'{self.rag_agent}', 'score': '10', 'tool_result': 'balahbalah'}, 'id': 'call_wTO40b9'}])

class DeciderAgentMock(DeciderAgent):
    def __init__(self, rag_agents, conversation_factory, expected: str):
        super().__init__(rag_agents, conversation_factory)
        self.expected_ragagent = expected
    def _create_initial_responder(self, prompt: str | None = None) -> ResponderWithRetries:
        runnable = InitialResponder(self.expected_ragagent)
        validator = PydanticToolsParser(tools=[ChooseRagAgent])
        return ResponderWithRetries(
            runnable=runnable,
            validator=validator,
        )
    def _create_revise_responder(self, prompt: str | None = None) -> ResponderWithRetries:
        runnable = ReviseResponder(self.expected_ragagent)
        validator = PydanticToolsParser(tools=[ReviseRagAgent])
        return ResponderWithRetries(
            runnable=runnable,
            validator=validator
        )

def createDeciderAgent(expected_rag_agent: str):
    dbAgent = MagicMock()
    dbAgent.mode = RagAgentModeEnum.KG
    dbAgent.get_description = MagicMock(return_value = "mock database agent")
    dbAgent.generate_responses = MagicMock(return_value=[
        ('{"bp.name": "balahbalah"}', {'cypher_query': "balahbalah"}),
        ('{"bp.name": "balahbalah"}', {'cypher_query': "balahbalah"})
    ])
    vectorstoreAgent = MagicMock()
    vectorstoreAgent.mode = RagAgentModeEnum.VectorStore
    vectorstoreAgent.get_description = MagicMock(return_value="mock vector store")
    vectorstoreAgent.generate_responses = MagicMock(return_value=[
        "mock content 1",
        "mock content 2"
    ])
    blastAgent = MagicMock()
    blastAgent.mode = RagAgentModeEnum.API_BLAST
    blastAgent.get_description = MagicMock(return_value="mock blast api")
    blastAgent.generate_responses = MagicMock(return_value=[
        "blast search result 1",
        "blast search result 2",
    ])
    oncokbAgent = MagicMock()
    oncokbAgent.mode = RagAgentModeEnum.API_ONCOKB
    oncokbAgent.get_description = MagicMock(return_value="mock oncokb api")
    oncokbAgent.generate_responses = MagicMock(return_value=[
        "oncokb search result 1",
        "oncokb search result 2",
    ])
    return DeciderAgentMock(
        rag_agents=[dbAgent, vectorstoreAgent, blastAgent, oncokbAgent],
        conversation_factory=create_conversation,
        expected=expected_rag_agent,
    )

def test_deciderAgent_oncokb():
    deciderAgent = createDeciderAgent(str(RagAgentModeEnum.API_ONCOKB))
    result = deciderAgent.execute("What is the oncogenic potential of BRAF V600E mutation?")
    assert result.answer == "api_oncokb"

def test_deciderAgent_blast():
    deciderAgent = createDeciderAgent(str(RagAgentModeEnum.API_BLAST))
    result = deciderAgent.execute("What is the oncogenic potential of BRAF V600E mutation?")
    assert result.answer == "api_blast"

def test_deciderAgent_kg():
    deciderAgent = createDeciderAgent(str(RagAgentModeEnum.KG))
    result = deciderAgent.execute("What is the oncogenic potential of BRAF V600E mutation?")
    assert result.answer == "kg"

def test_deciderAgent_vectorstore():
    deciderAgent = createDeciderAgent(str(RagAgentModeEnum.VectorStore))
    result = deciderAgent.execute("What is the oncogenic potential of BRAF V600E mutation?")
    assert result.answer == "vectorstore"

