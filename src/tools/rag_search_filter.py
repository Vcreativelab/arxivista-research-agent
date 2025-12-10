# src/tools/rag_search_filter.py
# Retrieves relevant research content from the Pinecone vector database
# filtered by ArXiv ID. Returns unified output schema.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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


def rag_search_filter(query: str, arxiv_id: str, top_k: int = 6) -> Dict[str, Any]:
    """
    Retrieve relevant text chunks from Pinecone filtered by ArXiv ID.

    Args:
        query (str): The user's search query.
        arxiv_id (str): The specific ArXiv ID to filter results.
        top_k (int): Number of top matches to return.

    Returns:
        dict: Unified result schema.
    """
    vectorstore = get_vectorstore()
    metadata = {"query": query, "arxiv_id": arxiv_id, "top_k": top_k}

    try:
        results = vectorstore.similarity_search(query, k=top_k, filter={"arxiv_id": arxiv_id})
    except Exception as e:
        err = f"Pinecone filtered similarity_search failed: {e}"
        print(f"⚠️ {err}")
        return _wrap_response("rag_search_filter", False, [], metadata, error=err)

    if not results:
        msg = f"No vector matches found for ArXiv ID {arxiv_id} with query: '{query}'."
        print(f"ℹ️ {msg}")
        return _wrap_response("rag_search_filter", True, [{
            "content": msg,
            "title": "No Filtered RAG Results",
            "source": "system",
            "arxiv_id": arxiv_id
        }], metadata)

    normalized = []
    for r in results:
        normalized.append({
            "content": getattr(r, "page_content", "") or "",
            "title": r.metadata.get("title", "Untitled Paper"),
            "source": r.metadata.get("source", "arxiv"),
            "arxiv_id": r.metadata.get("arxiv_id", arxiv_id)
        })

    return _wrap_response("rag_search_filter", True, normalized, metadata)
