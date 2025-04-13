# It Retrieves relevant research content from the Pinecone vector database.

from langchain_pinecone import PineconeVectorStore
from src.config import embeddings, INDEX_NAME  # Import shared config


def rag_search(query: str, top_k: int = 5):
    """
    Retrieves relevant papers from Pinecone vector index.

    Args:
        query (str): The search query.
        top_k (int): Number of top matches to return.

    Returns:
        list: List of dictionaries with keys "text" and "source".
    """
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    results = vectorstore.similarity_search(query, k=top_k)
    return [{"text": r.page_content, "source": r.metadata.get("source", "")} for r in results]
