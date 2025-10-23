# src/tools/rag_search.py
# Retrieves relevant research papers from the Pinecone vector database
# using semantic similarity search.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain_community.vectorstores import Pinecone
from src.config import embeddings, INDEX_NAME


@st.cache_resource
def get_vectorstore():
    """Initialize and cache Pinecone vectorstore across Streamlit pages."""
    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)


def rag_search(query: str, top_k: int = 5):
    """
    Retrieve semantically relevant papers from Pinecone index.

    Args:
        query (str): The user's research query.
        top_k (int): Number of top matching text chunks to return.

    Returns:
        list[dict]: Each dict contains 'text', 'title', 'source', and 'arxiv_id'.
    """
    vectorstore = get_vectorstore()

    try:
        # Perform similarity search using OpenAI embeddings
        results = vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        print(f"⚠️ Pinecone search failed: {e}")
        return []

    # Return structured results for display in Streamlit
    return [
        {
            "text": r.page_content,
            "title": r.metadata.get("title", "Untitled Paper"),
            "source": r.metadata.get("source", "N/A"),
            "arxiv_id": r.metadata.get("arxiv_id", "N/A"),
        }
        for r in results
    ]
