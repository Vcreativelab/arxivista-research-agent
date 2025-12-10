# src/tools/web_search.py
# Safe & robust web search with retries and fallbacks.
# Returns unified output schema.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import List, Dict, Any
import time
import requests
from src.config import SERP_API_KEY


def _wrap_response(tool: str, success: bool, results: List[Dict[str, Any]], metadata: Dict[str, Any], error: str | None = None):
    return {
        "tool": tool,
        "success": success,
        "results": results,
        "metadata": metadata,
        "error": error
    }


def wikipedia_fallback(query: str) -> List[Dict[str, Any]]:
    """Fallback: try Wikipedia summary API. Always returns a list (possibly empty)."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '%20')}"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if "extract" in data:
                return [{
                    "title": data.get("title", "Wikipedia Result"),
                    "link": data.get("content_urls", {}).get("desktop", {}).get("page", "N/A"),
                    "snippet": data["extract"],
                    "source": "wikipedia"
                }]
    except Exception as e:
        print(f"⚠️ Wikipedia fallback failed: {e}")
    return []


def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Performs a web search using SerpAPI with retries and safe fallbacks.
    Guarantees unified output schema.

    Args:
        query (str): The search query.
        num_results (int): Number of results to return.

    Returns:
        dict: Unified result schema.
    """
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": SERP_API_KEY, "num": num_results}
    metadata = {"query": query, "num_results": num_results}

    # Retry with exponential backoff
    last_exception = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                organic = data.get("organic_results", []) or []
                if organic:
                    results = []
                    for r in organic[:num_results]:
                        results.append({
                            "title": r.get("title", "No Title"),
                            "link": r.get("link", "N/A"),
                            "snippet": r.get("snippet", "No snippet available."),
                            "source": r.get("source", "web")
                        })
                    return _wrap_response("web_search", True, results, metadata)
            # short backoff before next attempt
            time.sleep(0.8 * (attempt + 1))
        except Exception as e:
            last_exception = e
            print(f"⚠️ SerpAPI request failed (attempt {attempt+1}/3): {e}")
            time.sleep(0.8 * (attempt + 1))

    # Try Wikipedia fallback (guarantees a list result)
    wiki_results = wikipedia_fallback(query)
    if wiki_results:
        return _wrap_response("web_search", True, wiki_results, metadata)

    # Final safe fallback (empty structured message)
    err_msg = f"SerpAPI failed after retries. Last error: {last_exception}"
    print(f"⚠️ {err_msg}")
    fallback = [{
        "title": "Web search unavailable",
        "link": "N/A",
        "snippet": f"Neither SerpAPI nor Wikipedia could answer the query: '{query}'.",
        "source": "system"
    }]
    return _wrap_response("web_search", False, fallback, metadata, error=str(last_exception))
