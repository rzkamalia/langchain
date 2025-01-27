import openai
import os

from dotenv import find_dotenv, load_dotenv

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph, START


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

model = ChatOpenAI(model="gpt-4o")

class State(MessagesState):
    summary: str

def call_model(state: State):
    
    # get summary if it exists
    summary = state.get("summary", "")

    # if there is summary, then we add it
    if summary:
        system_message = f"""Summary of conversation earlier: {summary}"""
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    
    # we get any existing summary
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"""This is summary of the conversation to date: {summary}\n\n
            Extend the summary by taking into account the new messages above:"""
        )
    else:
        summary_message = """Create a summary of the conversation above:"""

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_continue(state: State):
    """Return the next node to execute.
    """
    messages = state["messages"]
    
    # if there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # otherwise we can just end
    return END

graph = StateGraph(State)
graph.add_node("conversation", call_model)
graph.add_node(summarize_conversation)

graph.add_edge(START, "conversation")
graph.add_conditional_edges("conversation", should_continue)
graph.add_edge("summarize_conversation", END)

memory = MemorySaver()
graph = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "2"}}

# mode = update
# for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Yoona")]}, config, stream_mode="updates"):
#     chunk['conversation']["messages"].pretty_print()

# mode = values
input_message = HumanMessage(content="hi! I'm Yoona")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    for m in event['messages']:
        m.pretty_print()
    print("---"*25)