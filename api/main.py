from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.orchestrator import run_pipeline
from agents.state import AgentState

app = FastAPI(
    title="IntelliAgent",
    description="Multi-agent financial analysis API",
    version="1.0.0"
)


class AnalysisRequest(BaseModel):
    query: str


class AnalysisResponse(BaseModel):
    query: str
    ticker: str
    signal: str
    confidence: float
    research_summary: str
    final_report: str


@app.get("/")
def root():
    return {"status": "running", "service": "IntelliAgent"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    """
    Run the full multi-agent pipeline on a financial query.
    
    Example query: "Analyze AAPL stock for a short-term trade"
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result: AgentState = run_pipeline(request.query)
        return AnalysisResponse(
            query=result["query"],
            ticker=result.get("ticker", "UNKNOWN"),
            signal=result.get("signal", "HOLD"),
            confidence=result.get("confidence", 0.5),
            research_summary=result.get("research_output", ""),
            final_report=result.get("final_report", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}
