import openai
import os

from dotenv import find_dotenv, load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, MessagesState, StateGraph, START


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-4o")

def filter_messages(state: MessagesState):
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):    
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("filter", filter_messages)
graph.add_node("chat_model", chat_model_node)
graph.add_edge(START, "filter")
graph.add_edge("filter", "chat_model")
graph.add_edge("chat_model", END)
graph = graph.compile()

messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Yoona", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Yoona", id="4"))

output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
