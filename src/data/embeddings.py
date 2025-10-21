# src/data/embeddings.py

import fitz  # PyMuPDF for PDF text extraction
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from src.config import embeddings, INDEX_NAME


@st.cache_resource
def get_vectorstore():
    """Initialize and cache Pinecone vectorstore."""
    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def process_pdf(pdf_path: str, metadata: dict):
    """Split PDF text into manageable chunks with metadata."""
    # Increase chunk size for fewer API calls & faster embedding generation
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    text = extract_text_from_pdf(pdf_path)
    chunks = splitter.split_text(text)
    chunk_metadata = [metadata] * len(chunks) if metadata else [{}] * len(chunks)
    return chunks, chunk_metadata


def create_embeddings(pdf_paths: list[str], metadata_list: list[dict] | None = None):
    """
    Generate embeddings from PDFs and store them in Pinecone.

    Args:
        pdf_paths (list): List of local PDF file paths.
        metadata_list (list of dict, optional): Metadata per PDF (e.g., title, arxiv_id).
    """
    vectorstore = get_vectorstore()

    all_texts, all_metadata = [], []

    # Process PDFs concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_pdf, pdf_paths, metadata_list or [{}] * len(pdf_paths)))

    for texts, meta in results:
        all_texts.extend(texts)
        all_metadata.extend(meta)

    if not all_texts:
        print("⚠️ No text chunks to embed.")
        return

    print(f"🚀 Storing {len(all_texts)} text chunks in Pinecone...")
    vectorstore.add_texts(all_texts, metadatas=all_metadata)
    print("✅ Embeddings successfully stored in Pinecone.")
