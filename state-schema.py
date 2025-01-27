from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph, START


class State(TypedDict):
    foo: int

def node_1(state):
    print("Node 1")
    return {"foo": state['foo'] + 1}

def node_2(state):
    print("Node 2")
    return {"foo": state['foo'] + 1}

def node_3(state):
    print("Node 3")
    return {"foo": state['foo'] + 1}

graph = StateGraph(State)
graph.add_node("node_1", node_1)
graph.add_node("node_2", node_2)
graph.add_node("node_3", node_3)

graph.add_edge(START, "node_1")
graph.add_edge("node_1", "node_2")
graph.add_edge("node_1", "node_3")
graph.add_edge("node_2", END)
graph.add_edge("node_3", END)

graph = graph.compile()
# result = graph.invoke({"foo" : 1})
# print(result)

# the result will be:
# langgraph.errors.InvalidUpdateError: At key 'foo': Can receive only one value per step. Use an Annotated key to handle multiple values.

class State(TypedDict):
    foo: Annotated[list[int], add]

def node_1(state):
    print("Node 1")
    return {"foo": [state['foo'][-1] + 1]}

def node_2(state):
    print("Node 2")
    return {"foo": [state['foo'][-1] + 1]}

def node_3(state):
    print("Node 3")
    return {"foo": [state['foo'][-1] + 1]}

graph_new  = StateGraph(State)
graph_new.add_node("node_1", node_1)
graph_new.add_node("node_2", node_2)
graph_new.add_node("node_3", node_3)

graph_new.add_edge(START, "node_1")
graph_new.add_edge("node_1", "node_2")
graph_new.add_edge("node_1", "node_3")
graph_new.add_edge("node_2", END)
graph_new.add_edge("node_3", END)

graph_new = graph_new.compile()
result_new = graph_new.invoke({"foo" : [1]})
print(result_new)

# the result will be:
# {'foo': [1, 2, 3, 3]}
