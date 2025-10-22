# pages/1_Configure_ArXiv.py
# It Fetches research papers from ArXiv.

import streamlit as st
from src.data.dataset import fetch_arxiv_papers

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Configure ArXiv", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>⚙️ <span style='color:#6C63FF;'>Configure</span> ArXiv</h1>",
    unsafe_allow_html=True
)
st.write("Use the controls below to select a category and fetch recent research papers from ArXiv.")

# ---------------- Styling (matches other pages) ----------------
st.markdown("""
    <style>
        :root {
            --primary-color-light: #6C63FF;
            --secondary-color-light: #00C9A7;
            --primary-color-dark: #8E7CFF;
            --secondary-color-dark: #00EBC7;
        }

        @media (prefers-color-scheme: light) {
            div.stButton > button:first-child {
                background: linear-gradient(90deg, var(--primary-color-light), var(--secondary-color-light));
                color: white;
            }
        }

        @media (prefers-color-scheme: dark) {
            div.stButton > button:first-child {
                background: linear-gradient(90deg, var(--primary-color-dark), var(--secondary-color-dark));
                color: white;
            }
        }

        div.stButton > button:first-child {
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.4em;
            transition: all 0.3s ease;
        }

        div.stButton > button:first-child:hover {
            filter: brightness(1.1);
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Cache Layer ----------------
@st.cache_data
def cached_fetch_arxiv(category, paper_count):
    """Cached wrapper for ArXiv paper fetching."""
    return fetch_arxiv_papers(category, paper_count)

# ---------------- UI Controls ----------------
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    category = st.selectbox("📂 Select an ArXiv category:", ["cs.AI", "cs.LG", "cs.CL", "cs.NE", "cs.CV"])
    paper_count = st.slider("📄 Number of papers to fetch:", 1, 50, 10)
    fetch_pressed = st.button("Fetch Papers")

# ---------------- Main Logic ----------------
if fetch_pressed:
    with st.spinner("🔍 Fetching papers from ArXiv..."):
        papers = cached_fetch_arxiv(category, paper_count)
        if not papers:
            st.error("❌ Failed to fetch papers. Please try again later.")
        else:
            st.session_state["arxiv_papers"] = papers
            st.success(f"✅ Successfully fetched {len(papers)} papers from ArXiv.")

            # Display paper previews
            for paper in papers:
                st.markdown(f"**{paper['title']}**  \n*{', '.join(paper['authors'])}*  \n{paper['summary']}")
                st.markdown("---")

            st.success(f"✅ Successfully fetched {len(papers)} papers from ArXiv.\n\n"
                        "**Next:** Move to the **Build Knowledge** page to process and embed them."
            ) 
