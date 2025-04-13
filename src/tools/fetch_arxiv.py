# It Fetches relevant research papers from ArXiv using its API.

import requests
from bs4 import BeautifulSoup
import time

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def fetch_arxiv(query: str, max_results: int = 10, retries: int = 3, timeout: int = 10):
    """Fetches ArXiv papers matching the query with retry logic."""

    params = {"search_query": query, "start": 0, "max_results": max_results}

    for attempt in range(retries):
        try:
            response = requests.get(ARXIV_API_URL, params=params, timeout=timeout)
            response.raise_for_status()

            # Parse response if successful
            soup = BeautifulSoup(response.text, "xml")
            papers = []
            for entry in soup.find_all("entry"):
                papers.append({
                    "title": entry.title.text,
                    "authors": [author.text for author in entry.find_all("author")],
                    "summary": entry.summary.text,
                    "pdf_url": entry.id.text.replace("abs", "pdf") + ".pdf"
                })
            return papers

        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout error on attempt {attempt + 1}/{retries}. Retrying...")
            time.sleep(2)

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch ArXiv data: {e}")
            break

    return []