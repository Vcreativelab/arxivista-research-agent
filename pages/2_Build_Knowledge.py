# pages/2_Build_Knowledge.py
# It Processes the papers and indexes them in Pinecone.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


# ---------------- Sequential Navigation Guard ----------------
### Hard stop if Page 1 not completed
if "arxiv_papers" not in st.session_state:
    st.warning("⚠️ No papers fetched yet. Go to **Configure ArXiv** first.")
    st.stop()


# ---------------- Duplicate Build Guard ----------------
if st.session_state.get("vectorstore_ready"):
    st.success("✅ Knowledge base already built.")
    st.page_link(
        "pages/3_Ask_Research_Agent.py",
        label="Go to Research Agent",
        icon="🤖"
    )
    st.stop()


# ---------------- Processing Lock Init ----------------
### Initialize ONLY
if "processing_running" not in st.session_state:
    st.session_state.processing_running = False


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
    r = requests.get(url)
    try:
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.JSONDecodeError:
        return None

processing_animation = load_lottie_url(
    "https://lottie.host/5c704725-3696-45d6-8f23-264cc68f79d1/2zs2F5H8uY.json"
)


# ---------------- Main UI ----------------
st.markdown("### 🗂️ Process and index the downloaded research papers into Pinecone")
# Get a preview of fetched papers
with st.expander("📄 Papers to be indexed"):
    for p in st.session_state["arxiv_papers"]:
        st.markdown(f"**{p['title']}**")
        st.caption(", ".join(p["authors"]))

col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    process_pressed = st.button(
        "🚀 Process & Index Papers",
        disabled=st.session_state.processing_running
    )


# ---------------- Processing Pipeline ----------------
### Correct lock usage    
if process_pressed and not st.session_state.processing_running:
    
    st.session_state.processing_running = True
    
    # Animation placeholders
    animation_placeholder = st.empty()
    text_placeholder = st.empty()

    with animation_placeholder.container():
        st_lottie(processing_animation, height=200, key="processing")
        text_placeholder.write("⚙️ Downloading PDFs, processing text, and indexing into Pinecone...")

    # ---------------- NEW PIPELINE ----------------
    # process_papers returns (pdf_paths, metadata_list)
    pdf_paths, metadata_list = process_papers(st.session_state["arxiv_papers"])

    if not pdf_paths:
        animation_placeholder.empty()
        text_placeholder.empty()
        st.session_state.processing_running = False
        st.error("❌ No PDFs could be processed. Check internet connection or try fewer papers.")
        st.stop()

    # create_embeddings now requires both pdf paths and their metadata
    st.session_state["vectorstore_ready"] = False
    create_embeddings(pdf_paths, metadata_list)
    # Mark vectorstore ready
    st.session_state["vectorstore_ready"] = True

    # Save ONLY successfully indexed papers
    st.session_state["indexed_papers"] = list(
        st.session_state["arxiv_papers"]
    )
    st.session_state["vectorstore_ready"] = True
    # ----------------------------------------------

    # Clear animation
    animation_placeholder.empty()
    text_placeholder.empty()
    
    st.session_state.processing_running = False

    st.success(
        f"✅ Successfully processed {len(pdf_paths)} papers and indexed their content into Pinecone!\n\n"
        "➡️ Next step: "
    )
    st.page_link("pages/3_Ask_Research_Agent.py", label="Ask Research Agent", icon="3️⃣")
