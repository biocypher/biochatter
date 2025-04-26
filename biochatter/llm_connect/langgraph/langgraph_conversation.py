"""Module for LangGraph-based conversation implementation.

This module provides a conversation implementation based on LangGraph, which
enables more explicit state management and control flow for conversations with
LLM providers, including tool usage and response correction capabilities.
"""

import asyncio
import json
import logging
import urllib.parse
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

import nltk
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from biochatter._image import encode_image, encode_image_from_url
from biochatter.llm_connect.available_models import TOOL_CALLING_MODELS
from biochatter.llm_connect.conversation import (
    ADDITIONAL_TOOL_RESULT_INTERPRETATION_PROMPT,
    GENERAL_TOOL_RESULT_INTERPRETATION_PROMPT,
    TOOL_RESULT_INTERPRETATION_PROMPT,
    TOOL_USAGE_PROMPT,
    Conversation,
)

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """Type definition for the LangGraph conversation state."""

    messages: Annotated[list[BaseMessage], add_messages]
    ca_messages: Annotated[list[BaseMessage], add_messages]
    original_question: str
    context_statements: list[str]
    tool_results: list[dict[str, Any]] | None
    final_response: str | None
    needs_correction: bool
    split_correction: bool
    explain_tool_result: bool
    tool_call_mode: Literal["auto", "text"]
    available_tools: list[Callable]
    corrections: list[str] | None
    token_usage: dict[str, int]
    model_name: str
    mcp: bool
    has_tool_calls: bool
    general_instructions_tool_interpretation: str
    additional_instructions_tool_interpretation: str


class LangGraphConversation(Conversation):
    """Conversation class using LangGraph for state management and control flow.

    This class implements the Conversation interface using LangGraph to manage
    the conversation state and control flow, including tool usage and response
    correction capabilities.
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
        use_ragagent_selector: bool = False,
        tools: list[Callable] = None,
        tool_call_mode: Literal["auto", "text"] = "auto",
        mcp: bool = False,
        additional_tools_instructions: str = None,
    ) -> None:
        """Initialize the LangGraphConversation class.

        Args:
            model_name (str): The name of the model to use.
            model_provider (str): The provider of the model.
            prompts (dict): Dictionary of prompts for different conversation stages.
            correct (bool): Whether to use the correcting agent.
            split_correction (bool): Whether to correct sentence by sentence.
            use_ragagent_selector (bool): Whether to use the RAG agent selector.
            tools (List[Callable]): List of tools to make available.
            tool_call_mode (Literal["auto", "text"]): How to handle tool calls.
            mcp (bool): Whether to use MCP mode.
            additional_tools_instructions (str): Additional instructions for tool usage.

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
            use_ragagent_selector=use_ragagent_selector,
            tools=tools,
            tool_call_mode=tool_call_mode,
            mcp=mcp,
            additional_tools_instructions=additional_tools_instructions,
        )

        self.model_provider = model_provider
        self._graph = None
        self.general_instructions_tool_interpretation = GENERAL_TOOL_RESULT_INTERPRETATION_PROMPT
        self.additional_instructions_tool_interpretation = ADDITIONAL_TOOL_RESULT_INTERPRETATION_PROMPT

    def set_api_key(self, api_key: str | None = None, user: str | None = None) -> bool:
        """Set the API key for the model provider.
        
        If the key is valid, initialize the LLMs and build the graph.
        
        Args:
            api_key (str, optional): API key (loaded from env if None).
            user (str, optional): User identifier for API usage tracking.
            
        Returns:
            bool: True if API key is valid and setup successful.

        """
        self.user = user

        try:
            self.chat = init_chat_model(
                model=self.model_name,
                model_provider=self.model_provider,
                temperature=0,
            )

            self.ca_chat = init_chat_model(
                model=self.model_name,
                model_provider=self.model_provider,
                temperature=0,
            )

            # Bind tools if available
            if self.tools:
                self.bind_tools(self.tools)

            # Build the conversation graph
            self._build_graph()

            return True
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self._chat = None
            self._ca_chat = None
            return False

    def _build_graph(self) -> None:
        """Build the LangGraph for conversation flow.
        
        This method defines all the nodes and edges for the conversation graph.
        """
        # Define the graph
        builder = StateGraph(ConversationState)

        # Define nodes
        builder.add_node("inject_context", self._inject_context_node)
        builder.add_node("chatbot", self._chatbot_node)
        builder.add_node("tool_node", self._tool_node)
        builder.add_node("manual_tool_parser", self._manual_tool_parser_node)
        builder.add_node("tool_explainer", self._tool_explainer_node)
        builder.add_node("corrector", self._corrector_node)
        builder.add_node("aggregate_response", self._aggregate_response_node)

        # Define standard edges for the main flow
        builder.add_edge("inject_context", "chatbot")

        # Conditional routing from chatbot node
        builder.add_conditional_edges(
            "chatbot",
            self._route_from_chatbot,
            {
                "tool_node": "tool_node",
                "manual_parser": "manual_tool_parser",
                "corrector": "corrector",
                "complete": "aggregate_response",
            }
        )

        # Routes from tool nodes
        builder.add_conditional_edges(
            "tool_node",
            self._route_from_tool,
            {
                "explain": "tool_explainer",
                "chatbot": "chatbot",
                "complete": "aggregate_response",
            }
        )

        builder.add_conditional_edges(
            "manual_tool_parser",
            self._route_from_tool,
            {
                "explain": "tool_explainer",
                "chatbot": "chatbot",
                "complete": "aggregate_response",
            }
        )

        # Tool explainer routes
        builder.add_conditional_edges(
            "tool_explainer",
            self._route_from_tool_explainer,
            {
                "chatbot": "chatbot",
                "complete": "aggregate_response",
            }
        )

        # Corrector always goes to aggregate_response
        builder.add_edge("corrector", "aggregate_response")

        # Set entry point
        builder.set_entry_point("inject_context")

        # Compile the graph
        self._graph = builder.compile()

    def query(
        self,
        text: str,
        image_url: str | None = None,
        tools: list[Callable] | None = None,
        explain_tool_result: bool = False,
        additional_tools_instructions: str | None = None,
        general_instructions_tool_interpretation: str | None = None,
        additional_instructions_tool_interpretation: str | None = None,
        mcp: bool = False,
    ) -> tuple[str, dict | None, str | None]:
        """Query the LLM API using the user's query.
        
        Args:
            text (str): The user query.
            image_url (str, optional): URL of an image to include.
            tools (List[Callable], optional): Tools to make available for this query.
            explain_tool_result (bool): Whether to explain tool results.
            additional_tools_instructions (str, optional): Additional instructions for tools.
            general_instructions_tool_interpretation (str, optional): General prompt for tool interpretation.
            additional_instructions_tool_interpretation (str, optional): Additional prompt for tool interpretation.
            mcp (bool): Whether to use MCP mode.
            
        Returns:
            Tuple[str, Dict | None, str | None]: Response text, token usage, and correction if enabled.

        """
        if not self._graph:
            logger.error("Graph not initialized. Call set_api_key() first.")
            return "Error: Conversation not properly initialized.", None, None

        self.last_human_prompt = text
        self.mcp = mcp if mcp else self.mcp

        if additional_tools_instructions:
            self.additional_tools_instructions = additional_tools_instructions

        # Set tool interpretation prompts
        if general_instructions_tool_interpretation:
            self.general_instructions_tool_interpretation = general_instructions_tool_interpretation
        if additional_instructions_tool_interpretation:
            self.additional_instructions_tool_interpretation = additional_instructions_tool_interpretation

        # Prepare the message to add to conversation
        if not image_url:
            new_message = HumanMessage(content=text)
        else:
            # Handle image message
            parsed_url = urllib.parse.urlparse(image_url)
            local = not parsed_url.netloc

            if local or not parsed_url.netloc:
                image_url = f"data:image/jpeg;base64,{encode_image(image_url)}"
            else:
                image_url = f"data:image/jpeg;base64,{encode_image_from_url(image_url)}"

            new_message = HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            )

        # Create initial state
        initial_state = {
            "messages": self.messages + [new_message],
            "ca_messages": self.ca_messages.copy(),
            "original_question": text,
            "context_statements": [],
            "tool_results": [],
            "final_response": None,
            "needs_correction": self.correct,
            "split_correction": self.split_correction,
            "explain_tool_result": explain_tool_result,
            "tool_call_mode": self.tool_call_mode,
            "available_tools": (self.tools or []) + (tools or []),
            "corrections": [],
            "token_usage": {"total": 0},
            "model_name": self.model_name,
            "mcp": self.mcp,
            "has_tool_calls": False,
            "general_instructions_tool_interpretation": self.general_instructions_tool_interpretation,
            "additional_instructions_tool_interpretation": self.additional_instructions_tool_interpretation,
        }

        # Run the graph
        try:
            final_state = self._graph.invoke(initial_state)

            # Update the conversation state
            self.messages = final_state["messages"]
            self.ca_messages = final_state["ca_messages"]

            # Extract results to return
            response = final_state["final_response"]
            token_usage = final_state["token_usage"]

            # Get corrections if available
            corrections = None
            if final_state.get("corrections") and len(final_state["corrections"]) > 0:
                corrections = "\n".join(final_state["corrections"])

            return response, token_usage, corrections

        except Exception as e:
            logger.error(f"Error running graph: {e}")
            return f"Error: {e!s}", None, None

    # Graph Node Implementations

    def _inject_context_node(self, state: ConversationState) -> dict:
        """Node for injecting context from RAG agents.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            Dict: Updated state with injected context.

        """
        text = state["original_question"]
        statements = []

        if self.use_ragagent_selector:
            statements = self._inject_context_by_ragagent_selector(text)
        else:
            for agent in self.rag_agents:
                try:
                    docs = agent.generate_responses(text)
                    statements = statements + [doc[0] for doc in docs]
                except ValueError as e:
                    logger.warning(e)

        state["context_statements"] = statements

        # Add context to messages if statements available
        if statements and len(statements) > 0:
            prompts = self.prompts["rag_agent_prompts"]
            self.current_statements = statements

            system_messages = []
            for i, prompt in enumerate(prompts):
                # If last prompt, format the statements into the prompt
                if i == len(prompts) - 1:
                    system_messages.append(
                        SystemMessage(content=prompt.format(statements=statements))
                    )
                else:
                    system_messages.append(SystemMessage(content=prompt))

            # Add system messages to conversation
            state["messages"].extend(system_messages)

        return state

    def _chatbot_node(self, state: ConversationState) -> dict:
        """Node for primary LLM conversation.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            Dict: Updated state with LLM response.

        """
        available_tools = state["available_tools"]
        model_name = state["model_name"]

        chat = self.chat

        # Handle tools if available
        if model_name in TOOL_CALLING_MODELS and available_tools:
            chat = chat.bind_tools(available_tools)
        elif model_name not in TOOL_CALLING_MODELS and available_tools:
            # Create tool prompt for models without native tool calling
            tools_prompt = self._create_tool_prompt(
                tools=available_tools,
                additional_instructions=self.additional_tools_instructions,
                mcp=state["mcp"]
            )
            # Replace the last user message with the tools prompt
            messages = state["messages"].copy()
            messages[-1] = tools_prompt
            state["messages"] = messages

        try:
            response = chat.invoke(state["messages"])

            # Update token usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                state["token_usage"]["chatbot"] = response.usage_metadata.get("total_tokens", 0)
                state["token_usage"]["total"] = state["token_usage"].get("total", 0) + state["token_usage"]["chatbot"]

            # Check if response has tool calls
            has_tool_calls = bool(getattr(response, "tool_calls", None))
            state["has_tool_calls"] = has_tool_calls

            # Add response to messages
            state["messages"].append(response)

            return state

        except Exception as e:
            logger.error(f"Error in chatbot node: {e}")
            state["final_response"] = f"Error: {e!s}"
            return state

    def _route_from_chatbot(self, state: ConversationState) -> str:
        """Determine the next node after the chatbot node.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            str: Name of the next node.

        """
        # Get the latest AI message
        latest_message = next((msg for msg in reversed(state["messages"])
                           if isinstance(msg, AIMessage)), None)

        if not latest_message:
            return "complete"

        # Check if using native tool calls
        if state["has_tool_calls"] and state["model_name"] in TOOL_CALLING_MODELS:
            return "tool_node"

        # Check if using manual tool parsing
        if (state["model_name"] not in TOOL_CALLING_MODELS and
            state["available_tools"] and
            isinstance(latest_message.content, str) and
            (latest_message.content.startswith("{") or
            latest_message.content.strip().startswith("{"))):
            return "manual_parser"

        # Check if correction is needed
        if state["needs_correction"]:
            return "corrector"

        return "complete"

    def _tool_node(self, state: ConversationState) -> dict:
        """Node for executing native tool calls.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            Dict: Updated state with tool execution results.

        """
        # Get the latest AI message with tool calls
        latest_message = next((msg for msg in reversed(state["messages"])
                           if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)), None)

        if not latest_message or not latest_message.tool_calls:
            return state

        tool_results = []

        for tool_call in latest_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # Find matching tool
            tool_func = next((t for t in state["available_tools"] if t.name == tool_name), None)

            if tool_func:
                try:
                    # Execute the tool
                    if state["mcp"]:
                        loop = asyncio.get_event_loop()
                        tool_result = loop.run_until_complete(tool_func.ainvoke(tool_args))
                    else:
                        tool_result = tool_func.invoke(tool_args)

                    # Add result to state
                    tool_result_obj = {
                        "name": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                        "id": tool_call_id
                    }
                    tool_results.append(tool_result_obj)

                    # Add tool message to conversation
                    state["messages"].append(
                        ToolMessage(content=str(tool_result), name=tool_name, tool_call_id=tool_call_id)
                    )

                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {e!s}"
                    state["messages"].append(
                        ToolMessage(content=error_message, name=tool_name, tool_call_id=tool_call_id)
                    )
                    tool_results.append({
                        "name": tool_name,
                        "args": tool_args,
                        "error": str(e),
                        "id": tool_call_id
                    })

        state["tool_results"] = tool_results
        return state

    def _manual_tool_parser_node(self, state: ConversationState) -> dict:
        """Node for parsing and executing tools from model text response.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            Dict: Updated state with manual tool execution results.

        """
        # Get the latest AI message
        latest_message = next((msg for msg in reversed(state["messages"])
                           if isinstance(msg, AIMessage)), None)

        if not latest_message or not isinstance(latest_message.content, str):
            return state

        try:
            # Clean up the response to extract JSON
            content = latest_message.content
            content = content.replace('"""', "").replace("json", "").replace("`", "").strip()

            # Try to parse the JSON
            tool_call = json.loads(content)
            tool_name = tool_call.get("tool_name")

            if not tool_name:
                return state

            # Find the matching tool
            tool_func = next((t for t in state["available_tools"] if t.name == tool_name), None)

            if not tool_func:
                return state

            # Remove tool_name as it's not a parameter for the tool
            tool_args = tool_call.copy()
            del tool_args["tool_name"]

            # Execute the tool
            if state["mcp"]:
                loop = asyncio.get_event_loop()
                tool_result = loop.run_until_complete(tool_func.ainvoke(tool_args))
            else:
                tool_result = tool_func.invoke(tool_args)

            # Format the result
            formatted_result = f"Tool: {tool_name}\nArguments: {tool_args}\nTool result: {tool_result}"

            # Add to tool results
            state["tool_results"] = [{
                "name": tool_name,
                "args": tool_args,
                "result": tool_result
            }]

            # Replace the AI message with the tool result
            messages = state["messages"].copy()
            messages[-1] = AIMessage(content=formatted_result)
            state["messages"] = messages

            return state

        except Exception as e:
            logger.error(f"Error parsing manual tool call: {e}")
            return state

    def _route_from_tool(self, state: ConversationState) -> str:
        """Determine the next node after tool execution.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            str: Name of the next node.

        """
        if not state["tool_results"]:
            return "complete"

        # If tool result explanation is requested
        if state["explain_tool_result"]:
            return "explain"

        # If we need to continue conversation with the chatbot
        # This would be the case if the tool result needs further processing
        if state["tool_call_mode"] == "auto" and state["model_name"] in TOOL_CALLING_MODELS:
            return "chatbot"

        # Otherwise, we're done
        return "complete"

    def _tool_explainer_node(self, state: ConversationState) -> dict:
        """Node for explaining tool results.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            Dict: Updated state with tool result explanations.

        """
        if not state["tool_results"]:
            return state

        explanations = []

        for tool_result in state["tool_results"]:
            tool_name = tool_result["name"]
            result = tool_result.get("result", "")

            # Create the interpretation prompt
            prompt = TOOL_RESULT_INTERPRETATION_PROMPT.format(
                original_question=state["original_question"],
                tool_result=result,
                general_instructions=state["general_instructions_tool_interpretation"],
                additional_instructions=state["additional_instructions_tool_interpretation"]
            )

            # Get explanation from LLM
            explanation = self.chat.invoke(prompt)

            # Add to messages
            state["messages"].append(AIMessage(content=explanation.content))
            explanations.append({
                "tool": tool_name,
                "explanation": explanation.content
            })

            # Update token usage
            if hasattr(explanation, "usage_metadata") and explanation.usage_metadata:
                state["token_usage"]["tool_explainer"] = state["token_usage"].get("tool_explainer", 0) + explanation.usage_metadata.get("total_tokens", 0)
                state["token_usage"]["total"] = state["token_usage"].get("total", 0) + explanation.usage_metadata.get("total_tokens", 0)

        state["tool_explanations"] = explanations
        return state

    def _route_from_tool_explainer(self, state: ConversationState) -> str:
        """Determine the next node after tool explanation.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            str: Name of the next node.

        """
        # After explanation, we might want to continue with the chatbot
        if state["tool_call_mode"] == "auto" and state["model_name"] in TOOL_CALLING_MODELS:
            return "chatbot"

        return "complete"

    def _corrector_node(self, state: ConversationState) -> dict:
        """Node for correcting model responses.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            Dict: Updated state with corrections.

        """
        if not state["needs_correction"]:
            return state

        # Get the latest AI message for correction
        latest_message = next((msg for msg in reversed(state["messages"])
                           if isinstance(msg, AIMessage)), None)

        if not latest_message:
            return state

        msg_content = latest_message.content

        # Split correction if needed
        corrections = []

        if state["split_correction"]:
            nltk.download("punkt", quiet=True)
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            sentences = tokenizer.tokenize(msg_content)

            for sentence in sentences:
                correction, usage = self._get_correction(sentence, state["ca_messages"])
                if correction.lower() not in ["ok", "ok."]:
                    corrections.append(correction)

                # Update token usage
                if usage:
                    state["token_usage"]["corrector"] = state["token_usage"].get("corrector", 0) + usage
                    state["token_usage"]["total"] = state["token_usage"].get("total", 0) + usage
        else:
            correction, usage = self._get_correction(msg_content, state["ca_messages"])
            if correction.lower() not in ["ok", "ok."]:
                corrections.append(correction)

            # Update token usage
            if usage:
                state["token_usage"]["corrector"] = usage
                state["token_usage"]["total"] = state["token_usage"].get("total", 0) + usage

        state["corrections"] = corrections
        return state

    def _get_correction(self, msg: str, ca_messages: list[BaseMessage]) -> tuple[str, int]:
        """Get correction for a message.
        
        Args:
            msg (str): The message to correct.
            ca_messages (List[BaseMessage]): Correcting agent messages.
            
        Returns:
            Tuple[str, int]: Correction text and token usage.

        """
        messages = ca_messages.copy()
        messages.append(HumanMessage(content=msg))
        messages.append(SystemMessage(
            content="If there is nothing to correct, please respond with just 'OK', and nothing else!"
        ))

        try:
            response = self.ca_chat.invoke(messages)
            correction = response.content

            # Get token usage if available
            token_usage = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage = response.usage_metadata.get("total_tokens", 0)

            return correction, token_usage
        except Exception as e:
            logger.error(f"Error in correction: {e}")
            return f"Error in correction: {e!s}", 0

    def _aggregate_response_node(self, state: ConversationState) -> dict:
        """Node for aggregating the final response.
        
        Args:
            state (ConversationState): Current conversation state.
            
        Returns:
            Dict: Updated state with final response.

        """
        # Get the latest AI message
        latest_message = next((msg for msg in reversed(state["messages"])
                           if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None)), None)

        if not latest_message:
            state["final_response"] = "No response generated."
            return state

        final_response = latest_message.content
        state["final_response"] = final_response

        return state

    # Helper methods from Conversation class

    def _create_tool_prompt(self, tools: list[Callable], additional_instructions: str = None, mcp: bool = False) -> HumanMessage:
        """Create the tool prompt for models not supporting tool calling.
        
        Args:
            tools (List[Callable]): Available tools.
            additional_instructions (str, optional): Additional instructions.
            mcp (bool): Whether to use MCP mode.
            
        Returns:
            HumanMessage: Formatted tool prompt.

        """
        prompt_template = ChatPromptTemplate.from_template(TOOL_USAGE_PROMPT)
        tools_description = self._tool_formatter(tools, mcp=mcp)

        new_message = prompt_template.invoke(
            {
                "user_question": self.messages[-1].content,
                "tools": tools_description,
                "additional_tools_instructions": additional_instructions if additional_instructions else "",
            }
        )

        return new_message.messages[0]
