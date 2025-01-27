from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph, START


class State(TypedDict):
    state: Annotated[list, add]

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}

# example 1

graph = StateGraph(State)

graph.add_node("a", ReturnNodeValue("I'm A"))
graph.add_node("b", ReturnNodeValue("I'm B"))
graph.add_node("c", ReturnNodeValue("I'm C"))
graph.add_node("d", ReturnNodeValue("I'm D"))

graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", "c")
graph.add_edge("c", "d")
graph.add_edge("d", END)
graph = graph.compile()

# graph.invoke({"state": []})
# the result:
# Adding I'm A to []
# Adding I'm B to ["I'm A"]
# Adding I'm C to ["I'm A", "I'm B"]
# Adding I'm D to ["I'm A", "I'm B", "I'm C"]


# example 2: a case where one parallel path has more steps than the other one.

graph_1 = StateGraph(State)

graph_1.add_node("a", ReturnNodeValue("I'm A"))
graph_1.add_node("b", ReturnNodeValue("I'm B"))
graph_1.add_node("b2", ReturnNodeValue("I'm B2"))
graph_1.add_node("c", ReturnNodeValue("I'm C"))
graph_1.add_node("d", ReturnNodeValue("I'm D"))

graph_1.add_edge(START, "a")
graph_1.add_edge("a", "b")
graph_1.add_edge("a", "c")
graph_1.add_edge("b", "b2")
graph_1.add_edge(["b2", "c"], "d")
graph_1.add_edge("d", END)
graph_1 = graph_1.compile()

# graph_1.invoke({"state": []})
# the result:
# Adding I'm A to []
# Adding I'm B to ["I'm A"]
# Adding I'm C to ["I'm A"]
# Adding I'm B2 to ["I'm A", "I'm B", "I'm C"]
# Adding I'm D to ["I'm A", "I'm B", "I'm C", "I'm B2"]


# example 3: in example 2, C running before B2. if we want to custom the order like we want to run C after B2, we can use a custom reducer.

def sorting_reducer(left, right):
    """Combines and sorts the values in a list.
    """
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]
    
    return sorted(left + right, reverse=False)

class NewState(TypedDict):
    new_state: Annotated[list, sorting_reducer]

graph_2 = StateGraph(NewState)

graph_2.add_node("a", ReturnNodeValue("I'm A"))
graph_2.add_node("b", ReturnNodeValue("I'm B"))
graph_2.add_node("b2", ReturnNodeValue("I'm B2"))
graph_2.add_node("c", ReturnNodeValue("I'm C"))
graph_2.add_node("d", ReturnNodeValue("I'm D"))

graph_2.add_edge(START, "a")
graph_2.add_edge("a", "b")
graph_2.add_edge("a", "c")
graph_2.add_edge("b", "b2")
graph_2.add_edge(["b2", "c"], "d")
graph_2.add_edge("d", END)
graph_2 = graph_2.compile()

graph_2.invoke({"new_state": []})
# the result:
# Adding I'm A to []
# Adding I'm B to ["I'm A"]
# Adding I'm C to ["I'm A"]
# Adding I'm B2 to ["I'm A", "I'm B", "I'm C"]
# Adding I'm D to ["I'm A", "I'm B", "I'm C", "I'm B2"]
