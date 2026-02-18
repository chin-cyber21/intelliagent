from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    """Shared state passed between all agents in the pipeline."""
    query: str                          # Original user query 
    ticker: Optional[str]               # Extracted stock ticker
    research_output: Optional[str]      # Output from Research Agent
    analysis_output: Optional[str]      # Output from Analysis Agent
    rag_context: Optional[str]          # Retrieved documents from RAG
    final_report: Optional[str]         # Final synthesis output
    signal: Optional[str]               # Trading signal: BUY / SELL / HOLD
    confidence: Optional[float]         # Confidence score 0.0 - 1.0
    errors: List[str]                   # Any errors encountered during pipeline
