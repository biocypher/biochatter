# Reflexion Agent

While current LLMs have many capabilities, their outputs can be unstable at 
times. To stabilise responses and allow more complex agent workflows, we have
introduced a Reflexion Agent, allowing agents to reflect on their experiences,
score their output, and self-improve.

## Workflow

The workflow of a Reflexion Agent is composed of individual nodes that can
either generate or consume data as follows:

![ReflexionAgent workflow](images/reflexion-agent.png)

***draft***: in this node, an LLM is initially prompted to generate a specific text
and action. (mem <- AIMessage(...)).  

***execute tool***: this node executes a tool function based on an action/text
generated in the previous node. (mem <- ToolMessage(...)).  

***revise***: this node scores the output of the tool call and generates a
self-reflection to provide feedback aimed at improving the results. (mem <-
AIMessage(...))  

***evaluate***: this node assesses the quality of the generated outputs  

***memory***: a list of BaseMessage

## Usage

class `KGQueryReflexionAgent` provide the ability to generate Cypher query based on user's question.

To use `KGQueryReflexionAgent`,
1. We need to pass in connection arguments that enables to connect to the target graph database and a conversation factory, which can create an instance of GptConversation.
```
from biochatter.llm_connect import GptConversation
from biochatter.kg_langgraph_agent import KGQueryReflexionAgent
def create_conversation():
    conversation = GptConversation(model_name="gpt-3.5-turbo", prompts={})
    conversation.set_api_key("sk-xxx")
    return conversation

connection_args = {
    "host": "127.0.0.1",
    "port": "7687",
}

agent = KGQueryReflexionAgent(
    connection_args=connection_args,
    conversation_factory=create_conversation,
)
```
2. Generate Knowledge Graph prompts based on user's question with BioCypherPromptEngine, which provides nodes info, edges info and their properties based on user's question.
```
from biochatter.prompts import BioCypherPromptEngine
prompt_engine = BioCypherPromptEngine(
    model_name="gpt-3.5-turbo",
    schema_config_or_info_dict=schema_dict, # kg schema info inquired from graph database
    conversation_factory=create_conversation,
)
# generate kg prompts based on user's question
kg_prompts = prompt_engine.generate_query_prompts(question)

```
3. generate cypher query
```
cypher_query = agent.execute(question, kg_prompt)
```

## Implementation

To use the `ReflexionAgent` class, we need to implement the following abstract
methods:

1. _tool_function(self, state: List[BaseMessage]):   
execute tool function based on previous action/text and return ToolMessage

2. _create_initial_responder(self, prompt: Option[str]):  
create draft responder, which is used to generate the initial answer

3. _create_revise_responder(self, prompt: Optional[str]):  
create revise responder, which is used to score outputs and revise the answers

4. _log_step_message(self, step: int, node: str, output: BaseMessage):  
parse step message and generate logs

5. _log_final_result(self, output: BaseMessage):  
parse final result and generate logs

6. _parse_final_result(self, output: BaseMessage):  
parse final result

As an example, we use the `kg_langgraph_agent.py` implementation that can
reflect on the task of generating a knowledge graph query.
The `KGQueryReflexionAgent` derived from `ReflexionAgent` is the main class to
perform this task. In the `KGQueryReflexionAgent`, we have implemented the
abstract methods described above:

1. _tool_function(self, state: List[BaseMessage]):  
connect to kg database and query KG in draft/revise node

2. _create_initial_responder(self, prompt: Option[str]):  
create initial responder, which prompts LLM to generate the query

initial prompts:  
```
(
    "system",
    (
        "As a senior biomedical researcher and graph database expert, "
        f"your task is to generate '{query_lang}' queries to extract data from our graph database based on the user's question. "
        """Current time {time}. {instruction}"""
    ),
),
(
    "system",
    "Only generate query according to the user's question above.",
),
```

Initial answer schema:  

```
class GenerateQuery(BaseModel):
    """Generate the query."""

    answer: str = Field(
        description="Cypher query for graph database according to user's question."
    )
    reflection: str = Field(
        description="Your reflection on the initial answer, critique of what to improve"
    )
    search_queries: List[str] = Field(description="query for graph database")
```

3. _create_revise_responder(self, prompt: Optional[str]):  
create revise responder, which prompts LLM to score the outputs, reflects on the
outputs, and revises the current query

Revise prompts:  

```
"""
Revise your previous query using the query result and follow the guidelines:
1. If you consistently obtain empty results, please consider removing constraints such as relationship constraints to try to obtain a result.
2. You should use previous critique to improve your query.
3. Only generate a query without returning any other text.
"""
```

Revise answer schema:  

```
class ReviseQuery(GenerateQuery):
    """Revise your previous query according to your question."""

    revised_query: str = Field(description=REVISED_QUERY_"Revised query"DESCRIPTION)
    score: str = Field(description=(
    "the score for the query based on its query result"
    " and relevance to the user's question,"
    " with 0 representing the lowest score and 10 representing the highest score."))
```

4. _log_step_message(self, step: int, node: str, output: BaseMessage):  
parse message from current step and generate logs

5. _log_final_result(self, output: BaseMessage):  
parse final result and generate logs

6. _parse_final_result(self, output: BaseMessage):  
parse final result

7. _should_continue(self, state: List[BaseMessage]):  
assess output and determine if we can exit loop based on the following rules:  
  1). if loop steps are greater than limit (30 or user defined), exit  
  2). if score in previous revise node is greater than 7, exit  
  3). if query result in execute_tool node is not empty, exit  
