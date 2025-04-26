"""Defining the LangGraphConversation class."""

import asyncio
from collections.abc import Callable

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .state import ConversationState
from .templates import FIRST_RESPONDER_PROMPT_TEMPLATE, AnswerQuestion, TOOL_FORMULATOR_PROMPT


class LangGraphConversation:
    """A class for a conversation using LangGraph.
    """

    def __init__(self,
                 model:str,
                 model_provider:str,
                 checkpointer:None|MemorySaver|SqliteSaver,
                 tools:list[Callable]=None,
                 mcp:bool=False,
                 reflexion:bool=False):

        #store the flags
        self.mcp = mcp
        self.reflexion = reflexion

        #initialize the llm
        self.llm = init_chat_model(model_provider=model_provider,model=model)


        if tools is not None:
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

        #add the edge
        self.graph.add_conditional_edges("chatbot", self._tool_router,{"tool_node":"tool_node",END:END})
        self.graph.add_conditional_edges("start_node", self._first_router,{"first_responder_node":"first_responder_node","chatbot":"chatbot"})
        self.graph.add_edge("tool_node", "chatbot")

        #set the entry point
        self.graph.set_entry_point("start_node")
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    def _start_node(self,state:ConversationState)->ConversationState:
        return {
            "current_question": state["messages"][-1].content
        }

    def _first_responder_node(self,state:ConversationState)->ConversationState:
        first_responder_chain = FIRST_RESPONDER_PROMPT_TEMPLATE | self.llm.with_structured_output(AnswerQuestion)
        response = first_responder_chain.invoke(state)
        parametrization_chain = TOOL_FORMULATOR_PROMPT | self.llm_with_tools
        tool_calls = parametrization_chain.invoke({"current_question":state["current_question"],"search_queries":"\n".join(response.search_queries)})
        print(tool_calls)
        return {"tool_calls": tool_calls,"search_queries":response.search_queries}


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

    def get_graph(self):
        display(Image(self.app.get_graph().draw_mermaid_png()))

    def reset_state(self):
        self.app.reset_state()
