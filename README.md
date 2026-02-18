# IntelliAgent — Multi-Agent Financial Analysis

## Setup

```bash
git clone https://github.com/chin-cyber21/intelliagent
cd intelliagent
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

## Run the demo

```bash
python demo/run_demo.py
```

## Run the API

```bash
uvicorn api.main:app --reload
# API docs at http://localhost:8000/docs
```

## Example API call

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze AAPL stock for a short-term trade"}'
```

## How it works

The pipeline runs three agents in sequence via LangGraph:

1. **Research Agent** — retrieves context from the RAG system and summarizes key information about the query
2. **Analysis Agent** — performs structured financial analysis and generates a BUY/SELL/HOLD signal with confidence score
3. **Synthesis Agent** — combines both outputs into a final investment report

RAG uses hybrid retrieval — dense search via FAISS + BGE embeddings combined with BM25 sparse search for better coverage.

## Project structure

```
intelliagent/
├── agents/
│   ├── state.py            # Shared state definition for all agents
│   ├── orchestrator.py     # LangGraph pipeline + graph definition
│   ├── research_agent.py   # Research and information gathering
│   ├── analysis_agent.py   # Quantitative analysis + signal generation
│   └── synthesis_agent.py  # Final report synthesis
├── rag/
│   ├── indexer.py          # Document ingestion + FAISS index builder
│   └── retriever.py        # Hybrid dense + sparse retrieval
├── api/
│   └── main.py             # FastAPI endpoints
├── demo/
│   └── run_demo.py         # CLI demo
└── requirements.txt
```

Built by [Chirag Saraswat](https://github.com/chin-cyber21)
