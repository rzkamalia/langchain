from operator import add
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import END, StateGraph, START


class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]

class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state):
    """Get logs that contain a failure.
    """
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """Generate summary of failures.
    """
    failures = state["failures"]
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}

fa_graph = StateGraph(input=FailureAnalysisState, output=FailureAnalysisOutputState)
fa_graph.add_node("get_failures", get_failures)
fa_graph.add_node("generate_summary", generate_summary)

fa_graph.add_edge(START, "get_failures")
fa_graph.add_edge("get_failures", "generate_summary")
fa_graph.add_edge("generate_summary", END)

class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}

def send_to_slack():
    report = "foo bar baz"
    return {"report": report}

qs_graph = StateGraph(input=QuestionSummarizationState,output=QuestionSummarizationOutputState)
qs_graph.add_node("generate_summary", generate_summary)
qs_graph.add_node("send_to_slack", send_to_slack)

qs_graph.add_edge(START, "generate_summary")
qs_graph.add_edge("generate_summary", "send_to_slack")
qs_graph.add_edge("send_to_slack", END)

class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str                             # will only be generated in the FA sub-graph
    report: str                                 # will only be generated in the QS sub-graph
    processed_logs: Annotated[List[int], add]   # will be generated in BOTH sub-graphs

def clean_logs(state):
    raw_logs = state["raw_logs"]
    cleaned_logs = raw_logs    # data cleaning raw_logs -> docs 
    return {"cleaned_logs": cleaned_logs}

entry_graph = StateGraph(EntryGraphState)
entry_graph.add_node("clean_logs", clean_logs)
entry_graph.add_node("question_summarization", qs_graph.compile())
entry_graph.add_node("failure_analysis", fa_graph.compile())

entry_graph.add_edge(START, "clean_logs")
entry_graph.add_edge("clean_logs", "failure_analysis")
entry_graph.add_edge("clean_logs", "question_summarization")
entry_graph.add_edge("failure_analysis", END)
entry_graph.add_edge("question_summarization", END)

graph = entry_graph.compile()