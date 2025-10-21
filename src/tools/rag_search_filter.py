# src/tools/rag_search_filter.py
# Retrieves relevant research content from the Pinecone vector database
# filtered by ArXiv ID.

import streamlit as st
from langchain_community.vectorstores import Pinecone
from src.config import embeddings, INDEX_NAME


@st.cache_resource
def get_vectorstore():
    """Initialize and cache Pinecone vectorstore across Streamlit pages."""
    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)


def rag_search_filter(query: str, arxiv_id: str, top_k: int = 6):
    """
    Retrieve relevant text chunks from Pinecone filtered by ArXiv ID.

    Args:
        query (str): The user's search query.
        arxiv_id (str): The specific ArXiv ID to filter results.
        top_k (int): Number of top matches to return.

    Returns:
        list[dict]: Each dict contains text, source, title, and arxiv_id.
    """
    vectorstore = get_vectorstore()

    try:
        # Perform similarity search with metadata filter
        results = vectorstore.similarity_search(
            query,
            k=top_k,
            filter={"arxiv_id": arxiv_id}
        )
    except Exception as e:
        print(f"⚠️ Pinecone search failed: {e}")
        return []

    # Return structured response for UI use
    return [
        {
            "text": r.page_content,
            "source": r.metadata.get("source", "N/A"),
            "title": r.metadata.get("title", "Untitled Paper"),
            "arxiv_id": r.metadata.get("arxiv_id", arxiv_id)
        }
        for r in results
    ]
