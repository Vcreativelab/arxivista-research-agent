# It Retrieves relevant research content from the Pinecone vector database
# based on ArXive ID

from langchain_pinecone import PineconeVectorStore
from src.config import embeddings, INDEX_NAME  # Import shared config


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
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Perform filtered similarity search
    results = vectorstore.similarity_search(query, k=top_k, filter={"arxiv_id": arxiv_id})

    return [{"text": r.page_content, "source": r.metadata.get("source", "")} for r in results]
