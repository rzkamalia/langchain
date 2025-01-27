import openai
import os

from dotenv import find_dotenv, load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def subtraction(a: int, b: int) -> float:
    """Subtract a from b.

    Args:
        a: first int
        b: second int
    """
    return b - a

class Agent:

    def __init__(self, llm, tools, system):
        self.llm = llm
        self.tools = tools
        self.system = system

        self.llm_with_tools = self.llm.bind_tools(self.tools)

        graph = StateGraph(MessagesState)
        graph.add_node("assistant", self.assistant)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        graph.add_edge("tools", "assistant")
        self.graph = graph.compile()
    
    def tools_conditon(self, state: MessagesState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
    def assistant(self, state: MessagesState):
        return {"messages": [self.llm_with_tools.invoke([self.system] + state["messages"])]}

llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [add, subtraction]
prompt = """You are a helpful assistant tasked with performing arithmetic on a set of inputs."""
system = SystemMessage(content=prompt)
agent = Agent(llm=llm, tools=tools, system=system)

query = "Add 5 and 8. Subtract the output from 3."
messages = [HumanMessage(content=query)]
result = agent.graph.invoke({"messages": messages})
for m in result['messages']:
    m.pretty_print()