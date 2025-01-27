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

graph = StateGraph(MessagesState)

graph.add_node("assistant", assistant)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "assistant")
graph.add_conditional_edges(
    "assistant",
    tools_condition,
)
graph.add_edge("tools", "assistant")

memory = MemorySaver()
graph = graph.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "3"}}

initial_input = {"messages": HumanMessage(content="Adding 2 and 3")}
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

## browsing history

## we can use get_state to look at the current state of our graph, given the thread_id:
# print(graph.get_state({'configurable': {'thread_id': '1'}}))

## we can also browse the state history of our agent:
all_states = [s for s in graph.get_state_history(thread)]
# print(all_states)

## replaying

# to_replay = all_states[-2] # fo example, we want to re-play from the second message
# print(to_replay.values)
## the result: 
## {'messages': [HumanMessage(content='Adding 2 and 3', additional_kwargs={}, response_metadata={}, id='b138e68e-2657-4a3d-8d47-fe38697cd046')]} and

## to replay from here, we simply pass the config back to the agent.
## the graph knows that this checkpoint has aleady been executed.
## it just re-plays from this checkpoint.
# for event in graph.stream(None, to_replay.config, stream_mode="values"):
#     event['messages'][-1].pretty_print()

## forking

## if we wantg to run from the same step, but with different input

to_fork = all_states[-2]
# print(to_fork.values)
## the result:
## {'messages': [HumanMessage(content='Adding 2 and 3', additional_kwargs={}, response_metadata={}, id='acec92fb-0702-437c-8c8c-cfc1e6131125')]} 

fork_config = graph.update_state(
    to_fork.config,
    {"messages": [HumanMessage(content='Add 5 and 3', id=to_fork.values["messages"][0].id)]},
)
for event in graph.stream(None, fork_config, stream_mode="values"):
    event['messages'][-1].pretty_print()