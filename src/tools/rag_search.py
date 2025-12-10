# src/tools/rag_search.py
# Retrieves relevant research papers from the Pinecone vector database
# using semantic similarity search. Returns a unified tool output schema.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any
import streamlit as st
from langchain_community.vectorstores import Pinecone
from src.config import embeddings, INDEX_NAME


@st.cache_resource
def get_vectorstore():
    """Initialize and cache Pinecone vectorstore across Streamlit pages."""
    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)


def _wrap_response(tool: str, success: bool, results: List[Dict[str, Any]], metadata: Dict[str, Any], error: str | None = None):
    return {
        "tool": tool,
        "success": success,
        "results": results,
        "metadata": metadata,
        "error": error
    }


def rag_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Retrieve semantically relevant papers from Pinecone index and return unified output.

    Args:
        query (str): The user's research query.
        top_k (int): Number of top matching text chunks to return.

    Returns:
        dict: Unified result schema described above.
    """
    vectorstore = get_vectorstore()
    metadata = {"query": query, "top_k": top_k}

    try:
        results = vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        err = f"Pinecone similarity_search failed: {e}"
        print(f"⚠️ {err}")
        return _wrap_response("rag_search", False, [], metadata, error=err)

    if not results:
        msg = f"No vector matches found for query: '{query}'."
        print(f"ℹ️ {msg}")
        return _wrap_response("rag_search", True, [{
            "content": msg,
            "title": "No RAG Results",
            "source": "system",
            "arxiv_id": "N/A"
        }], metadata)

    # Normalize results
    normalized = []
    for r in results:
        normalized.append({
            "content": getattr(r, "page_content", "") or "",
            "title": r.metadata.get("title", "Untitled Paper"),
            "source": r.metadata.get("source", "arxiv"),
            "arxiv_id": r.metadata.get("arxiv_id", "N/A"),
        })

    return _wrap_response("rag_search", True, normalized, metadata)
