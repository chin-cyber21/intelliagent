from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.research_agent import research_agent
from agents.analysis_agent import analysis_agent
from agents.synthesis_agent import synthesis_agent
from rag.retriever import retrieve_context
import re


def extract_ticker(state: AgentState) -> AgentState:
    """Extract stock ticker from the query if present."""
    query = state["query"]
    # Simple ticker extraction - looks for uppercase 1-5 letter words
    matches = re.findall(r'\b[A-Z]{1,5}\b', query)
    ticker = matches[0] if matches else "UNKNOWN"
    return {**state, "ticker": ticker}


def retrieve_rag(state: AgentState) -> AgentState:
    """Retrieve relevant context from vector store before research."""
    context = retrieve_context(state["query"])
    return {**state, "rag_context": context}


def build_graph() -> StateGraph:
    """
    Build and compile the multi-agent LangGraph pipeline.
    
    Flow:
    extract_ticker → retrieve_rag → research → analysis → synthesis → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("extract_ticker", extract_ticker)
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("research", research_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("synthesis", synthesis_agent)

    # Define edges
    graph.set_entry_point("extract_ticker")
    graph.add_edge("extract_ticker", "retrieve_rag")
    graph.add_edge("retrieve_rag", "research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


def run_pipeline(query: str) -> AgentState:
    """Run the full multi-agent pipeline for a given query."""
    graph = build_graph()
    initial_state: AgentState = {
        "query": query,
        "ticker": None,
        "research_output": None,
        "analysis_output": None,
        "rag_context": None,
        "final_report": None,
        "signal": None,
        "confidence": None,
        "errors": []
    }
    result = graph.invoke(initial_state)
    return result
