# src/tools/rag_search.py

import streamlit as st
from langchain_community.vectorstores import Pinecone
from src.config import embeddings, INDEX_NAME


@st.cache_resource
def get_vectorstore():
    """Initialize and cache Pinecone vectorstore."""
    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)


def rag_search(query: str, top_k: int = 5):
    """
    Retrieves relevant papers from Pinecone vector index.

    Args:
        query (str): The search query.
        top_k (int): Number of top matches to return.

    Returns:
        list: List of dictionaries with keys "text", "title", "source", "arxiv_id".
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=top_k)

    return [
        {
            "text": r.page_content,
            "title": r.metadata.get("title", "Unknown"),
            "source": r.metadata.get("source", ""),
            "arxiv_id": r.metadata.get("arxiv_id", "")
        }
        for r in results
    ]
