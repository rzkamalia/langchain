import getpass 
import openai
import os

from dotenv import find_dotenv, load_dotenv
from operator import add
from typing import Annotated, TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-4o")

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
_set_env("TAVILY_API_KEY")

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, add]

def search_web(state):
    """Retrieve docs from web search.
    """
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state):
    """Retrieve docs from wikipedia.
    """
    search_docs = WikipediaLoader(
        query=state['question'], 
        load_max_docs=2
    ).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def generate_answer(state):
    """Node to answer a question.
    """

    context = state["context"]
    question = state["question"]

    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(
        question=question, 
        context=context
    )    
    
    answer = llm.invoke([SystemMessage(content=answer_instructions)] + [HumanMessage(content=f"Answer the question.")])
      
    return {"answer": answer}

graph = StateGraph(State)

graph.add_node("search_web", search_web)
graph.add_node("search_wikipedia", search_wikipedia)
graph.add_node("generate_answer", generate_answer)

graph.add_edge(START, "search_wikipedia")
graph.add_edge(START, "search_web")
graph.add_edge("search_wikipedia", "generate_answer")
graph.add_edge("search_web", "generate_answer")
graph.add_edge("generate_answer", END)
graph = graph.compile()

result = graph.invoke({"question": "Who is super junior?"})
print(result['answer'].content)
