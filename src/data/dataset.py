# src/data/dataset.py
# Fetch ArXiv papers and download PDFs. Produces consistent metadata for downstream tools.
import requests
import os
import re
import concurrent.futures
import time
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

ARXIV_API_URL = "http://export.arxiv.org/api/query"
PDF_DIR = "data/pdfs"


def _extract_arxiv_id_from_entry_id(entry_id: str) -> str:
    # Example entry_id: "http://arxiv.org/abs/2402.03300v1"
    m = re.search(r"\/abs\/([^\s\/]+)", entry_id)
    return m.group(1) if m else entry_id


def sanitize_filename(filename: str) -> str:
    filename = filename.replace('\n', ' ').strip()
    # keep only safe characters
    return re.sub(r'[<>:"/\\|?*]', '', filename)


def fetch_arxiv_papers(category: str, count: int = 10, retries: int = 3, timeout: int = 10):
    """
    Fetch metadata from arXiv for the given category.
    Returns a dict: {'papers': [ {title, authors, summary, pdf_url, arxiv_id, source} ], 'count': n}
    """
    query = f"cat:{category}"
    params = {"search_query": query, "start": 0, "max_results": count}
    for attempt in range(retries):
        try:
            resp = requests.get(ARXIV_API_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "xml")
            papers = []
            for entry in soup.find_all("entry"):
                entry_id = entry.id.text if entry.id else ""
                arxiv_id = _extract_arxiv_id_from_entry_id(entry_id)
                title = (entry.title.text or "").strip()
                authors = [a.text.strip() for a in entry.find_all("author")]
                summary = (entry.summary.text or "").strip()
                # arXiv gives 'id' like https://arxiv.org/abs/.... convert to pdf
                pdf_url = entry.id.text.replace("abs", "pdf") + ".pdf" if entry.id else ""
                papers.append({
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "pdf_url": pdf_url,
                    "arxiv_id": arxiv_id,
                    "source": "arxiv",
                })
            return {"papers": papers, "count": len(papers)}
        except requests.exceptions.Timeout:
            time.sleep(1 + attempt * 2)
            continue
        except requests.RequestException as e:
            print(f"❌ fetch_arxiv_papers failed: {e}")
            break
    return {"papers": [], "count": 0}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_pdf(paper: dict, timeout: int = 10):
    """
    Downloads a PDF for one paper dict.
    Returns tuple (pdf_path or None, metadata dict) -- metadata mirrors the paper input.
    """
    title = sanitize_filename(paper.get("title", paper.get("arxiv_id", "paper")))
    pdf_url = paper.get("pdf_url", "")
    if not pdf_url:
        print(f"⚠️ No pdf_url for {title}")
        return None, paper

    os.makedirs(PDF_DIR, exist_ok=True)
    # include arxiv id in filename to avoid collisions
    arxiv_id = paper.get("arxiv_id", "")
    safe_name = f"{title} - {arxiv_id}.pdf" if arxiv_id else f"{title}.pdf"
    pdf_path = os.path.join(PDF_DIR, safe_name)

    try:
        resp = requests.get(pdf_url, timeout=timeout)
        resp.raise_for_status()
        with open(pdf_path, "wb") as fd:
            fd.write(resp.content)
        print(f"✅ Downloaded: {safe_name}")
        # augment metadata with local path
        metadata = {
            **paper,
            "local_pdf_path": pdf_path,
            "downloaded": True
        }
        return pdf_path, metadata
    except requests.RequestException as e:
        print(f"❌ Error downloading {title}: {e}")
        metadata = {**paper, "downloaded": False}
        return None, metadata


def process_papers(arxiv_papers: list[dict]):
    """
    Downloads PDFs in parallel and returns:
      - pdf_paths: list of local file paths for successfully downloaded PDFs
      - metadata_list: list of metadata dicts aligned with pdf_paths (same length)
    """
    if not arxiv_papers:
        return [], []

    pdf_paths = []
    metadata_list = []

    # parallel download; results may contain None
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_pdf, p) for p in arxiv_papers]
        for f in concurrent.futures.as_completed(futures):
            try:
                pdf_path, meta = f.result()
            except Exception as e:
                print(f"❌ download job failed: {e}")
                continue
            if pdf_path:
                pdf_paths.append(pdf_path)
                metadata_list.append(meta)
            else:
                # keep failed metadata too if you want to inspect
                # but do not include missing pdfs
                print(f"⚠️ Skipped (download failed): {meta.get('title','unknown')}")

    return pdf_paths, metadata_list
