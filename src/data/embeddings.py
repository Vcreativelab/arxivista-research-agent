# It Handles document embeddings.

from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF for PDF text extraction
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import embeddings, INDEX_NAME  # Import shared config
import streamlit as st  # Use Streamlit caching

# Cache Pinecone connection (Avoids reloading every time)
@st.cache_resource
def get_pinecone_vector_store():
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


vector_store = get_pinecone_vector_store()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)


def process_pdf(pdf_path, metadata):
    """Processes a single PDF into text chunks with metadata."""
    # Larger chunks = fewer API calls <-(originally chunk_size=500, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = extract_text_from_pdf(pdf_path)
    chunks = text_splitter.split_text(text)

    # Ensure metadata is assigned correctly
    chunk_metadata = [metadata] * len(chunks) if metadata is not None else [{}] * len(chunks)
    return chunks, chunk_metadata


def create_embeddings(pdf_paths, metadata_list=None):
    """
    Processes PDFs, generates embeddings, and stores them in Pinecone.

    Args:
        pdf_paths (list): List of PDF file paths.
        metadata_list (list of dict, optional): List of metadata dictionaries corresponding to each PDF.

    Returns:
        None
    """
    all_texts, all_metadata = [], []

    # Process PDFs concurrently with `as_completed()` for faster execution
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf, metadata_list[i] if metadata_list else {}): pdf
                         for i, pdf in enumerate(pdf_paths)}

        for future in as_completed(future_to_pdf):
            texts, meta = future.result()
            all_texts.extend(texts)
            all_metadata.extend(meta)

    # Batch Upload to Pinecone
    batch_size = 100  # Upload in chunks of 100 for efficiency
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i: i + batch_size]
        batch_metadata = all_metadata[i: i + batch_size]

        # Store in Pinecone
        vector_store.add_texts(batch_texts, metadatas=batch_metadata)

    print(f"✅ Stored {len(all_texts)} document chunks in Pinecone successfully!")
