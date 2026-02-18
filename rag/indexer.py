import os
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths
INDEX_PATH = Path("rag/index/faiss.index")
DOCS_PATH = Path("rag/index/documents.pkl")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # Lightweight but strong model

# Sample financial documents for demo
SAMPLE_DOCUMENTS = [
    "Apple Inc (AAPL) reported Q4 2024 revenue of $94.9 billion, up 6% year-over-year. iPhone revenue grew 5.5% to $46.2 billion. Services revenue hit a record $24.2 billion, growing 12% YoY. The company returned $29 billion to shareholders through dividends and buybacks.",
    "NVIDIA Corporation (NVDA) continues to dominate the AI chip market with its H100 and A100 GPU lines. Data center revenue grew 279% YoY in Q2 2024 to $26.3 billion. The company's CUDA ecosystem creates strong moats against competitors AMD and Intel.",
    "Microsoft (MSFT) Azure cloud revenue grew 29% in Q1 2025, driven by AI workloads. Copilot integration across Office 365 suite added 6 million enterprise users in the quarter. Operating margins expanded to 44.6%.",
    "Tesla (TSLA) deliveries fell 4.8% in Q1 2024 to 386,810 vehicles, missing consensus estimates. Price cuts across the lineup squeezed automotive gross margins to 17.4%. Energy storage deployments hit a record 4.1 GWh, up 37% YoY.",
    "The Federal Reserve held interest rates steady at 5.25-5.50% in its September 2024 meeting. Fed officials project two rate cuts in 2025, down from three projected in June. Inflation remains above the 2% target at 2.7% PCE.",
    "Amazon (AMZN) AWS revenue grew 19% to $27.5 billion in Q3 2024. Operating income surged 56% to $17.4 billion. The company's AI services including Bedrock and SageMaker saw triple-digit growth in active customers.",
    "Alphabet (GOOGL) Google Search revenue grew 12% to $49.4 billion in Q3 2024. YouTube advertising revenue hit $8.9 billion, up 12% YoY. Google Cloud grew 35% to $11.4 billion, approaching profitability.",
    "Meta Platforms (META) revenue grew 19% to $40.6 billion in Q3 2024. Daily active users across the family of apps reached 3.29 billion. Reality Labs division reported a $4.4 billion operating loss as VR investments continue.",
]


def build_index(documents: list[str] = None) -> None:
    """
    Build FAISS index from documents.
    Uses BGE-small embeddings for fast, high-quality retrieval.
    """
    if documents is None:
        documents = SAMPLE_DOCUMENTS

    print(f"Building index from {len(documents)} documents...")

    # Load embedding model
    model = SentenceTransformer(EMBED_MODEL)

    # Split long documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))

    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index (inner product for cosine similarity with normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index and documents
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built and saved. {index.ntotal} vectors indexed.")


if __name__ == "__main__":
    build_index()
