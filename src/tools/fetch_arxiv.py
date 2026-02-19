# src/tools/fetch_arxiv.py
# Agent tool wrapper for ArXiv fetching

from src.data.dataset import fetch_arxiv_papers


def fetch_arxiv(query: str, max_results: int = 5):
    """
    Agent-accessible tool for fetching ArXiv papers.
    Uses dataset layer internally.
    """

    # Convert free-text query into ArXiv search_query format
    result = fetch_arxiv_papers(query, max_results)
    papers = result.get("papers", [])

    if not papers:
        return {
            "status": "no_results",
            "papers": []
        }

    return {
        "status": "success",
        "papers": papers
    }
