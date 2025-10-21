# It Processes the papers and indexes them in Pinecone.

import streamlit as st
st.set_page_config(page_title="Build Knowledge Base", layout="wide")  # ✅ Must be first command

from streamlit_lottie import st_lottie
import requests
from src.data.dataset import process_papers
from src.data.embeddings import create_embeddings

st.title("🛠 Build Knowledge Base")

# Cache Lottie animation, so it doesn't reload every time
@st.cache_resource
def load_lottie_url(url: str):
    """Fetches Lottie animation JSON from an online source and caches it."""
    r = requests.get(url)
    try:
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.JSONDecodeError:
        return None


# Load cached Lottie animation
processing_animation = load_lottie_url("https://lottie.host/5c704725-3696-45d6-8f23-264cc68f79d1/2zs2F5H8uY.json")

# Cache expensive functions like fetching APIs or initializing connections
@st.cache_resource
def get_pinecone_index():
    """Initializes Pinecone index only once."""
    from langchain_community.vectorstores import Pinecone
    from src.config import embeddings
    return Pinecone.from_existing_index(index_name="research-knowledge", embedding=embeddings)


# Only initialize Pinecone once instead of reloading on every page switch
vector_store = get_pinecone_index()

if "arxiv_papers" not in st.session_state:
    st.warning("No papers fetched yet. Go to 'Configure ArXiv' first.")
else:
    if st.button("Process & Index Papers"):
        # Create placeholders for animation and text
        animation_placeholder = st.empty()
        text_placeholder = st.empty()

        # Show Lottie animation
        with animation_placeholder.container():
            st_lottie(processing_animation, height=200, key="processing")
            text_placeholder.write("⚙️ Processing papers and indexing...")

        # Process papers & create embeddings
        pdf_paths = process_papers(st.session_state["arxiv_papers"])
        create_embeddings(pdf_paths)

        # Remove animation after completion
        animation_placeholder.empty()
        text_placeholder.empty()

        # Show success message
        st.success("✅ Papers processed and indexed successfully!")
