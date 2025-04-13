# It Fetches research papers from ArXiv.

# It Fetches research papers from ArXiv.

import streamlit as st
from src.data.dataset import fetch_arxiv_papers

st.set_page_config(page_title="Configure ArXiv", layout="wide")
st.title("📂 Configure ArXiv")
st.write("Select an ArXiv category and fetch AI research papers.")

# Cache fetched papers (to avoid re-fetching if already stored)
@st.cache_data
def cached_fetch_arxiv(category, paper_count):
    return fetch_arxiv_papers(category, paper_count)


# UI Elements
category = st.selectbox("Select an ArXiv category:", ["cs.AI", "cs.LG", "cs.CL", "cs.NE", "cs.CV"])
paper_count = st.slider("Number of papers to fetch:", 1, 50, 10)

if st.button("Fetch Papers"):
    with st.spinner("Fetching papers..."):
        papers = cached_fetch_arxiv(category, paper_count)
        if not papers:
            st.error("❌ Failed to fetch papers. Please try again later.")
        else:
            st.session_state["arxiv_papers"] = papers
            st.success(f"✅ Fetched {len(papers)} papers from ArXiv.")

            for paper in papers:
                st.markdown(f"**{paper['title']}**\n*{', '.join(paper['authors'])}*\n{paper['summary']}")

