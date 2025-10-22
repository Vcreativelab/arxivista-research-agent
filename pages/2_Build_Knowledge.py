# pages/2_Build_Knowledge.py
# It Processes the papers and indexes them in Pinecone.

import streamlit as st
from streamlit_lottie import st_lottie
import requests
from src.data.dataset import process_papers
from src.data.embeddings import create_embeddings

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Build Knowledge Base", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>📚 <span style='color:#6C63FF;'>Arxivista</span> Knowledge Builder</h1>",
    unsafe_allow_html=True
)

# ---------------- Gradient Styling ----------------
st.markdown("""
    <style>
        :root {
            --primary-color-light: #6C63FF;
            --secondary-color-light: #00C9A7;
            --text-color-light: #000000;
            --primary-color-dark: #8E7CFF;
            --secondary-color-dark: #00EBC7;
            --text-color-dark: #FFFFFF;
        }

        @media (prefers-color-scheme: light) {
            div.stButton > button:first-child {
                background: linear-gradient(90deg, var(--primary-color-light), var(--secondary-color-light));
                color: var(--text-color-light);
            }
        }

        @media (prefers-color-scheme: dark) {
            div.stButton > button:first-child {
                background: linear-gradient(90deg, var(--primary-color-dark), var(--secondary-color-dark));
                color: var(--text-color-dark);
            }
        }

        div.stButton > button:first-child {
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            filter: brightness(1.1);
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Lottie Animation Loader ----------------
@st.cache_resource
def load_lottie_url(url: str):
    """Fetches Lottie animation JSON from an online source and caches it."""
    r = requests.get(url)
    try:
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.JSONDecodeError:
        return None

processing_animation = load_lottie_url(
    "https://lottie.host/5c704725-3696-45d6-8f23-264cc68f79d1/2zs2F5H8uY.json"
)

# ---------------- Pinecone Index Loader ----------------
@st.cache_resource
def get_pinecone_index():
    """Initializes Pinecone index only once."""
    from langchain_community.vectorstores import Pinecone
    from src.config import embeddings
    return Pinecone.from_existing_index(index_name="research-knowledge", embedding=embeddings)

# Initialize Pinecone once
vector_store = get_pinecone_index()

# ---------------- Main UI ----------------
if "arxiv_papers" not in st.session_state:
    st.warning("⚠️ No papers fetched yet. Go to **Configure ArXiv** first.")
else:
    st.markdown("### 🗂️ Process and index the downloaded research papers into Pinecone")

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        process_pressed = st.button("🚀 Process & Index Papers")

    if process_pressed:
        # Create placeholders for animation and text
        animation_placeholder = st.empty()
        text_placeholder = st.empty()

        with animation_placeholder.container():
            st_lottie(processing_animation, height=200, key="processing")
            text_placeholder.write("⚙️ Processing papers and indexing...")

        # Process papers & create embeddings
        pdf_paths = process_papers(st.session_state["arxiv_papers"])
        create_embeddings(pdf_paths)

        # Clear animation
        animation_placeholder.empty()
        text_placeholder.empty()

        # Success message + navigation hint
        st.success(
            f"✅ Papers processed and indexed successfully!\n\n"
            "➡️ Next step: "
        )
        st.page_link("pages/3_Ask_Research_Agent.py", label="Ask Research Agent", icon="3️⃣")
