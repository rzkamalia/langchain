import openai
import os

from dotenv import find_dotenv, load_dotenv 

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
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

tools = [add, subtraction]
model = ChatOpenAI(model="gpt-4o")
llm_with_tools = model.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def human_feedback(state: MessagesState):
    pass

graph = StateGraph(MessagesState)

graph.add_node("assistant", assistant)
graph.add_node("tools", ToolNode(tools))
graph.add_node("human_feedback", human_feedback)

graph.add_edge(START, "human_feedback")
graph.add_edge("human_feedback", "assistant")
graph.add_conditional_edges(
    "assistant",
    tools_condition,
)
graph.add_edge("tools", "human_feedback")

memory = MemorySaver()
graph = graph.compile(interrupt_before=["human_feedback"], checkpointer=memory)

thread = {"configurable": {"thread_id": "4"}}

initial_input = {"messages": HumanMessage(content="Adding 2 and 3")}
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

# edit by user
user_input = input("Tell me how you want to update the state: ")
graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
