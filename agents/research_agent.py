from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.state import AgentState


RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial research agent. Your job is to gather and summarize 
relevant information about a stock or financial query. Focus on:
- Recent news and developments
- Company fundamentals (revenue, earnings, growth)
- Industry trends and competitive position
- Key risks and opportunities

Be factual and concise. Do not give trading advice — just present the information."""),
    ("human", "Research the following: {query}\n\nTicker: {ticker}\n\nAdditional context from knowledge base:\n{rag_context}")
])


def research_agent(state: AgentState) -> AgentState:
    """
    Research Agent — gathers and summarizes information about the query.
    Runs first in the pipeline before analysis.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    chain = RESEARCH_PROMPT | llm

    result = chain.invoke({
        "query": state["query"],
        "ticker": state.get("ticker", "Unknown"),
        "rag_context": state.get("rag_context", "No additional context available.")
    })

    return {
        **state,
        "research_output": result.content
    }
