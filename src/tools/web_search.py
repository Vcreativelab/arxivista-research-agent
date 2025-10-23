# It Performs a web search using SerpAPI (Google Search API) or another provider.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import requests
from src.config import SERP_API_KEY  # Import shared config


def web_search(query: str, num_results: int = 5):
    """
    Performs a web search using SerpAPI.

    Args:
        query (str): The search query.
        num_results (int): Number of results to return.

    Returns:
        list: List of dictionaries with keys "title", "link", and "snippet".
    """
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("organic_results", [])
        return [{"title": r["title"], "link": r["link"], "snippet": r.get("snippet", "")} for r in results]
    else:
        return []

