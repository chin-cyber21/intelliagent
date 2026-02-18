from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.state import AgentState


ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a quantitative financial analysis agent. Given research about a stock,
your job is to perform structured analysis and generate a trading signal.

Your output must always include:
1. Fundamental Analysis — valuation, growth, profitability
2. Technical Outlook — momentum, trend, key levels  
3. Risk Assessment — key downside risks
4. Signal — one of: BUY / SELL / HOLD
5. Confidence — a score from 0.0 to 1.0 representing your conviction

Format your signal clearly at the end like:
SIGNAL: BUY
CONFIDENCE: 0.75"""),
    ("human", "Based on this research, provide your analysis:\n\n{research_output}")
])


def analysis_agent(state: AgentState) -> AgentState:
    """
    Analysis Agent — takes research output and produces structured
    financial analysis with a trading signal and confidence score.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    chain = ANALYSIS_PROMPT | llm

    result = chain.invoke({
        "research_output": state.get("research_output", "No research available.")
    })

    content = result.content

    # Parse signal and confidence from output
    signal = "HOLD"
    confidence = 0.5

    for line in content.split("\n"):
        if line.startswith("SIGNAL:"):
            signal = line.replace("SIGNAL:", "").strip()
        if line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                confidence = 0.5

    return {
        **state,
        "analysis_output": content,
        "signal": signal,
        "confidence": confidence
    }
