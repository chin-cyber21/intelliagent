from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.state import AgentState


SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior portfolio manager synthesizing inputs from a research team.
You receive research findings and quantitative analysis, and your job is to produce a 
final, clean investment report that a fund manager could act on.

Structure your report as:
- Executive Summary (2-3 sentences)
- Key Findings (bullet points)
- Risk Factors
- Final Recommendation with rationale

Write clearly and professionally. No fluff."""),
    ("human", """Query: {query}
Ticker: {ticker}

Research Summary:
{research_output}

Quantitative Analysis:
{analysis_output}

Signal: {signal} | Confidence: {confidence}

Write the final investment report.""")
])


def synthesis_agent(state: AgentState) -> AgentState:
    """
    Synthesis Agent — final agent in the pipeline.
    Combines research + analysis into a clean investment report.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    chain = SYNTHESIS_PROMPT | llm

    result = chain.invoke({
        "query": state["query"],
        "ticker": state.get("ticker", "Unknown"),
        "research_output": state.get("research_output", ""),
        "analysis_output": state.get("analysis_output", ""),
        "signal": state.get("signal", "HOLD"),
        "confidence": state.get("confidence", 0.5)
    })

    return {
        **state,
        "final_report": result.content
    }
