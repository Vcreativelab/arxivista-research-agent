# src/tools/web_search.py
# Safe & robust web search with retries and fallbacks.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import requests
from src.config import SERP_API_KEY


# Use Wikipedia API when SerpAPI fails
def wikipedia_fallback(query: str):
    """Fallback: try Wikipedia summary API."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '%20')}"
        resp = requests.get(url, timeout=6)

        if resp.status_code == 200:
            data = resp.json()
            if "extract" in data:
                return [{
                    "title": data.get("title", "Wikipedia Result"),
                    "link": data.get("content_urls", {}).get("desktop", {}).get("page", "N/A"),
                    "snippet": data["extract"]
                }]
    except Exception as e:
        print(f"⚠️ Wikipedia fallback failed: {e}")

    return None



def web_search(query: str, num_results: int = 5):
    """
    Performs a web search using SerpAPI with retries and safe fallbacks.

    Returns:
        list[dict]: Always returns a *non-empty list* containing
        title, link, snippet.
    """

    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results,
    }

    # Retry 3 times with exponential backoff
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=6)

            # SerpAPI success
            if response.status_code == 200:
                data = response.json()
                organic = data.get("organic_results", [])

                if organic:
                    return [
                        {
                            "title": r.get("title", "No Title"),
                            "link": r.get("link", "N/A"),
                            "snippet": r.get("snippet", "No snippet available."),
                        }
                        for r in organic[:num_results]
                    ]

            # Short wait before retry
            time.sleep(0.8 * (attempt + 1))

        except Exception as e:
            print(f"⚠️ SerpAPI request failed (attempt {attempt+1}/3): {e}")
            time.sleep(0.8 * (attempt + 1))

    # Fallback to Wikipedia
    wiki = wikipedia_fallback(query)
    if wiki:
        return wiki

    # Guaranteed safe fallback
    return [{
        "title": "Web search unavailable",
        "link": "N/A",
        "snippet": f"Neither SerpAPI nor Wikipedia could answer the query: '{query}'."
    }]
