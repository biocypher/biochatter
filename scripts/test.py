import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

msg = [
    SystemMessage(content="Be nice"),
    HumanMessage(content="Hi there"),
]

result = chat.generate([msg])
print(result.generations[0][0])
