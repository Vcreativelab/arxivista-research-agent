# It Handles fetching PDFs and saving them.

import requests
import os
import re
import concurrent.futures
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def fetch_arxiv_papers(category: str, count: int):
    query = f"cat:{category}"
    params = {"search_query": query, "start": 0, "max_results": count}
    response = requests.get(ARXIV_API_URL, params=params)

    papers = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "xml")
        for entry in soup.find_all("entry"):
            papers.append({
                "title": entry.title.text,
                "authors": [author.text for author in entry.find_all("author")],
                "summary": entry.summary.text,
                "pdf_url": entry.id.text.replace("abs", "pdf") + ".pdf"
            })
    return papers


def sanitize_filename(filename):
    """Removes invalid characters from a filename."""
    filename = filename.replace('\n', ' ')  # Replace newline with space
    return re.sub(r'[<>:"/\\|?*]', '', filename)  # Remove illegal characters


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_pdf(paper):
    """Downloads a single PDF file with retry and timeout handling."""
    title = sanitize_filename(paper["title"])
    pdf_url = paper["pdf_url"]
    pdf_path = os.path.join("data/pdfs", f"{title}.pdf")

    os.makedirs("data/pdfs", exist_ok=True)

    try:
        response = requests.get(pdf_url, timeout=5)  # Set timeout to prevent hanging
        response.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        print(f"✅ Downloaded: {title}")
        return pdf_path

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {title}: {e}")
        return None


def process_papers(arxiv_papers):
    """Processes ArXiv papers, downloads PDFs in parallel, and returns PDF paths."""
    pdf_paths = []

    # Use ThreadPoolExecutor to parallelize downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(download_pdf, arxiv_papers))

    # Filter out None values (failed downloads)
    pdf_paths = [path for path in results if path is not None]

    return pdf_paths
