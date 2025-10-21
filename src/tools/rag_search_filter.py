# src/tools/rag_search_filter.py
# Retrieves relevant research content from the Pinecone vector database
# filtered by ArXiv ID.

import streamlit as st
from langchain_community.vectorstores import Pinecone
from src.config import embeddings, INDEX_NAME


@st.cache_resource
def get_vectorstore():
    """Initialize and cache Pinecone vectorstore."""
    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)


def rag_search_filter(query: str, arxiv_id: str, top_k: int = 6):
    """
    Retrieves relevant papers from Pinecone vector index filtered by ArXiv ID.

    Args:
        query (str): The search query.
        arxiv_id (str): The specific ArXiv ID to filter results.
        top_k (int): Number of top matches to return.

    Returns:
        list: List of dictionaries with keys "text" and "source".
    """
    vectorstore = get_vectorstore()

    # Perform filtered similarity search (only returns chunks from that paper)
    results = vectorstore.similarity_search(
        query,
        k=top_k,
        filter={"arxiv_id": arxiv_id}
    )

    return [
        {
            "text": r.page_content,
            "source": r.metadata.get("source", ""),
            "title": r.metadata.get("title", "Unknown"),
            "arxiv_id": r.metadata.get("arxiv_id", arxiv_id)
        }
        for r in results
    ]
