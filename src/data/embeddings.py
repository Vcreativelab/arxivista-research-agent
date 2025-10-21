# src/data/embeddings.py

import fitz
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    text = extract_text_from_pdf(pdf_path)
    chunks = splitter.split_text(text)
    chunk_metadata = [metadata] * len(chunks) if metadata else [{}] * len(chunks)
    return chunks, chunk_metadata


def create_embeddings(pdf_paths: list[str], metadata_list: list[dict] | None = None):
    """Generate embeddings from PDFs and store them in Pinecone."""
    vectorstore = get_vectorstore()
    all_texts, all_metadata = [], []

    # Parallel processing
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_pdf, pdf_paths, metadata_list or [{}] * len(pdf_paths)))

    for texts, meta in results:
        all_texts.extend(texts)
        all_metadata.extend(meta)

    if not all_texts:
        print("⚠️ No text chunks to embed.")
        return

    print(f"🚀 Preparing to store {len(all_texts)} text chunks in Pinecone...")

    # ✅ Process embeddings in smaller batches
    batch_size = 80
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_metadata = all_metadata[i:i + batch_size]

        try:
            vectorstore.add_texts(batch_texts, metadatas=batch_metadata)
        except Exception as e:
            print(f"⚠️ Error embedding batch {i // batch_size + 1}: {e}")

    print("✅ All text chunks successfully embedded and stored in Pinecone!")
