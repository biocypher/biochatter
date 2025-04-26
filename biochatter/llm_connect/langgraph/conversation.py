"""Defining the LangGraphConversation class."""

import asyncio
from collections.abc import Callable

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain.schema import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .state import ConversationState
from .templates import (
    FIRST_INSTRUCTION,
    REVISE_INSTRUCTIONS,
    TOOL_FORMULATOR_PROMPT,
    AnswerQuestion,
    #ReviseAnswer,
    get_actor_prompt_template,
)


class LangGraphConversation:
    """A class for a conversation using LangGraph.
    """

    def __init__(self,
                 model:str,
                 model_provider:str,
                 checkpointer:None|MemorySaver|SqliteSaver,
                 tools:list[Callable]=None,
                 mcp:bool=False,
                 reflexion:bool=False,
                 max_iterations:int=2):

        #store the flags
        self.mcp = mcp
        self.reflexion = reflexion

        #store the max iterations
        self.max_iterations = max_iterations
        #initialize the llm
        self.llm = init_chat_model(model_provider=model_provider,model=model)


        if self.reflexion and tools is not None:
            self.tool_node = ToolNode(tools,messages_key="tool_calls")
            self.llm_with_tools = self.llm.bind_tools(tools=tools)
            self._tool_formatter(tools)
        elif tools is not None:
            self.tool_node = ToolNode(tools)
            self.llm_with_tools = self.llm.bind_tools(tools=tools)
        else:
            self.llm_with_tools = None

        #initialize the checkpointer
        self.checkpointer = checkpointer

        #initialize the graph defining the nodes
        self.graph = StateGraph(ConversationState)
        self.graph.add_node("chatbot", self._chatbot_node)
        self.graph.add_node("tool_node", self.tool_node)
        self.graph.add_node("start_node", self._start_node)
        self.graph.add_node("first_responder_node", self._first_responder_node)
        self.graph.add_node("revisor_node", self._revisor_node)

        #add the edge
        self.graph.add_conditional_edges("chatbot", self._tool_router,{"tool_node":"tool_node",END:END})
        self.graph.add_conditional_edges("start_node", self._first_router,{"first_responder_node":"first_responder_node","chatbot":"chatbot"})
        self.graph.add_conditional_edges("tool_node", self._tool_return_router,{"revisor_node":"revisor_node","chatbot":"chatbot"})
        self.graph.add_edge("first_responder_node", "tool_node")
        #self.graph.add_edge("revisor_node", "tool_node")
        self.graph.add_conditional_edges("revisor_node", self._should_continue_router,{END:END,"tool_node":"tool_node"})
        #set the entry point
        self.graph.set_entry_point("start_node")
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    def _start_node(self,state:ConversationState)->ConversationState:
        return {
            "current_question": state["messages"][-1].content
        }

    def _first_responder_node(self,state:ConversationState)->ConversationState:
        first_responder_prompt = get_actor_prompt_template(FIRST_INSTRUCTION, self.tools_description)
        first_responder_chain = first_responder_prompt | self.llm.with_structured_output(AnswerQuestion)
        response = first_responder_chain.invoke(state)

        #first_answer = AIMessage(content="Tool plan:\n"+response.tool_plan+"\n\nAnswer:\n"+response.answer+f"\n\nReferences:\n{response.references}")
        first_answer = AIMessage(content="Tool plan:\n"+response.tool_plan+"\n\nAnswer:\n"+response.answer)
        first_revision = HumanMessage(content=f"Here is a revision:\nMissing: {response.reflection.missing}\nSuperfluous: {response.reflection.superfluous}")

        parametrization_chain = TOOL_FORMULATOR_PROMPT | self.llm_with_tools
        # Format search_queries as:
        # [1] query_1
        # [2] query_2
        formatted_queries = "\n".join([f"[{i+1}] {q}" for i, q in enumerate(response.search_queries)])
        tool_calls = parametrization_chain.invoke({
            "current_question": state["current_question"],
            "search_queries": formatted_queries
        })
        return {"tool_calls": [tool_calls],"search_queries":response.search_queries,"messages":[first_answer,first_revision],"iteration":0}


    def _revisor_node(self,state:ConversationState)->ConversationState:
        state["messages"].extend(state["tool_calls"])
        revisor_prompt = get_actor_prompt_template(REVISE_INSTRUCTIONS, self.tools_description)
        revising_chain = revisor_prompt | self.llm.with_structured_output(AnswerQuestion)
        response = revising_chain.invoke(state)

        #answer = AIMessage(content="Tool plan:\n"+response.tool_plan+"\n\nAnswer:\n"+response.answer+f"\n\nReferences:\n{response.references}")
        answer = AIMessage(content="Tool plan:\n"+response.tool_plan+"\n\nAnswer:\n"+response.answer)
        revision = HumanMessage(content=f"Here is a revision:\nMissing: {response.reflection.missing}\nSuperfluous: {response.reflection.superfluous}")

        parametrization_chain = TOOL_FORMULATOR_PROMPT | self.llm_with_tools
        formatted_queries = "\n".join([f"[{i+1}] {q}" for i, q in enumerate(response.search_queries)])
        tool_calls = parametrization_chain.invoke({
            "current_question": state["current_question"],
            "search_queries": formatted_queries
        })
        return {"tool_calls": [tool_calls],"search_queries":response.search_queries,"iteration":state["iteration"]+1,"messages":[answer,revision]}

    def _chatbot_node(self,state:ConversationState)->ConversationState:
        if self.llm_with_tools is not None:
            response = self.llm_with_tools.invoke(state["messages"])
        else:
            response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def _first_router(self,state:ConversationState)->str:
        if self.reflexion:
            return "first_responder_node"
        return "chatbot"

    def _tool_router(self,state:ConversationState)->str:
        last_message = state["messages"][-1]

        if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
            return "tool_node"
        return END

    def _tool_return_router(self,state:ConversationState)->str:
        if self.reflexion:
            return "revisor_node"
        return "chatbot"

    def _should_continue_router(self,state:ConversationState)->str:
        if self.reflexion and state["iteration"] < self.max_iterations:
            return "tool_node"
        return END

    def invoke(self,message:str,config:None | dict=None)->ConversationState:

        if config is None:
            config = {"configurable": {
                    "thread_id": 1
                        }}

        if self.mcp:
            loop = asyncio.get_running_loop()
            graph_result = loop.run_until_complete(self.app.ainvoke({
                "messages": [HumanMessage(content=message)]
            }, config=config))
            return graph_result
        return self.app.invoke({
            "messages": [HumanMessage(content=message)]
        }, config=config)

    def _tool_formatter(self,tools:list[Callable])->str:
        tools_description = ""

        for idx, tool in enumerate(tools):
            tools_description += f"<tool_{idx}>\n"
            tools_description += f"Tool name: {tool.name}\n"
            tools_description += f"Tool description: {tool.description}\n"
            if self.mcp:
                schema = str(tool.tool_call_schema)
                schema = schema.replace("{", "{{").replace("}", "}}")
                tools_description += f"Tool call schema:\n {schema}\n"
            #else:
            #    tools_description += f"Tool call schema:\n {tool.args}\n"
            tools_description += f"</tool_{idx}>\n"
        self.tools_description = tools_description

    def get_graph(self):
        display(Image(self.app.get_graph().draw_mermaid_png()))

    def reset_state(self):
        self.app.reset_state()
