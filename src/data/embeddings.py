# src/data/embeddings.py

from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF for PDF text extraction
from concurrent.futures import ThreadPoolExecutor
from src.config import embeddings, INDEX_NAME  # Import shared config
import streamlit as st


@st.cache_resource
def get_vector_store():
    """Cache Pinecone vector store across Streamlit pages."""
    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)


vector_store = get_vector_store()


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)


def process_pdf(pdf_path, metadata):
    """Processes a single PDF into text chunks with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = extract_text_from_pdf(pdf_path)
    chunks = text_splitter.split_text(text)
    chunk_metadata = [metadata] * len(chunks) if metadata else [{}] * len(chunks)
    return chunks, chunk_metadata


def create_embeddings(pdf_paths, metadata_list=None):
    """Processes PDFs, generates embeddings, and stores them in Pinecone."""
    all_texts, all_metadata = [], []

    # Process PDFs concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_pdf, pdf_paths, metadata_list or [{}] * len(pdf_paths)))

    for texts, meta in results:
        all_texts.extend(texts)
        all_metadata.extend(meta)

    if not all_texts:
        print("⚠️ No text chunks to embed.")
        return

    print(f"Embedding and storing {len(all_texts)} chunks in Pinecone...")

    # Add texts directly (LangChain handles embeddings + upsert)
    vector_store.add_texts(all_texts, metadatas=all_metadata)

    print("✅ Embeddings successfully stored in Pinecone.")

