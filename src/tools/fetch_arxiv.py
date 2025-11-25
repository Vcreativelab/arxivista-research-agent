# src/data/embeddings.py
# Create text chunks from PDFs and push embeddings into Pinecone with consistent metadata.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
import streamlit as st
from src.config import embeddings, INDEX_NAME

PDF_CHUNK_SIZE = 1200
PDF_CHUNK_OVERLAP = 100
BATCH_SIZE = 80


@st.cache_resource
def get_vectorstore():
    """Initialize and cache Pinecone vectorstore."""
    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        with fitz.open(pdf_path) as doc:
            pages_text = [p.get_text("text") for p in doc]
            return "\n".join(pages_text)
    except Exception as e:
        print(f"⚠️ Failed to extract text from {pdf_path}: {e}")
        return ""


def process_pdf(pdf_path: str, metadata: dict) -> Tuple[List[str], List[dict]]:
    """
    Split PDF text into chunks and attach metadata for each chunk.
    Returns (chunks, metadatas_for_chunks)
    """
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return [], []

    splitter = RecursiveCharacterTextSplitter(chunk_size=PDF_CHUNK_SIZE, chunk_overlap=PDF_CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    # Create per-chunk metadata by extending the given metadata
    chunk_meta = []
    base_meta = metadata.copy() if metadata else {}
    # ensure required keys exist
    base_meta.setdefault("source", base_meta.get("source", "arxiv"))
    base_meta.setdefault("title", base_meta.get("title", "Unknown"))
    base_meta.setdefault("arxiv_id", base_meta.get("arxiv_id", "N/A"))
    base_meta.setdefault("local_pdf_path", base_meta.get("local_pdf_path", None))

    for i, _ in enumerate(chunks):
        m = dict(base_meta)
        m["chunk_index"] = i
        chunk_meta.append(m)
    return chunks, chunk_meta


def create_embeddings(pdf_paths: List[str], metadata_list: Optional[List[dict]] = None):
    """
    Generate embeddings from PDFs and store them in Pinecone.
    pdf_paths: list of local PDF paths
    metadata_list: same-length list of metadata dicts aligned with pdf_paths
    """
    if not pdf_paths:
        print("⚠️ No pdfs to process.")
        return

    vectorstore = get_vectorstore()
    all_texts = []
    all_metadata = []

    # metadata_list must align with pdf_paths
    metadata_list = metadata_list or [{}] * len(pdf_paths)
    if len(metadata_list) != len(pdf_paths):
        # If mismatch, fill missing with empty metadata
        metadata_list = (metadata_list + [{}] * len(pdf_paths))[:len(pdf_paths)]

    # parallel processing of pdfs
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_pdf, pdf_paths, metadata_list))

    for texts, metas in results:
        if texts and metas:
            all_texts.extend(texts)
            all_metadata.extend(metas)

    if not all_texts:
        print("⚠️ No text chunks were created; nothing to embed.")
        return

    print(f"🚀 Preparing to store {len(all_texts)} text chunks in Pinecone...")

    # store in batches to avoid timeouts
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        batch_metas = all_metadata[i:i + BATCH_SIZE]
        try:
            vectorstore.add_texts(batch_texts, metadatas=batch_metas)
        except Exception as e:
            print(f"⚠️ Error embedding batch {i // BATCH_SIZE + 1}: {e}")

    print("✅ All text chunks successfully embedded and stored in Pinecone!")
