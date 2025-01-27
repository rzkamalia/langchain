from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, StateGraph, START


class State(TypedDict):
    input: str

def step_1(state: State) -> State:
    print("---Step 1---")
    return state

def step_2(state: State) -> State:
    if len(state['input']) > 8:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
    
    print("---Step 2---")
    return state

def step_3(state: State) -> State:
    print("---Step 3---")
    return state

graph = StateGraph(State)
graph.add_node("step_1", step_1)
graph.add_node("step_2", step_2)
graph.add_node("step_3", step_3)

graph.add_edge(START, "step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", "step_3")
graph.add_edge("step_3", END)

memory = MemorySaver()
graph = graph.compile(checkpointer=memory)

thread_config = {"configurable": {"thread_id": "1"}}

initial_input = {"input": "hello world"}
for event in graph.stream(initial_input, thread_config, stream_mode="values"):
    print(event)

graph.update_state(
    thread_config,
    {"input": "hi Yoon"},
)

for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)