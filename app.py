# It Handles navigation and main UI.

import streamlit as st

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.sidebar.title("🔍 AI Research Assistant")
st.sidebar.page_link("pages/1_Configure_ArXiv.py", label="📂 Configure ArXiv")
st.sidebar.page_link("pages/2_Build_Knowledge.py", label="🛠 Build Knowledge Base")
st.sidebar.page_link("pages/3_Ask_Research_Agent.py", label="🤖 Ask Research Agent")

st.title("🔮 AI-Powered Research Assistant")
st.write("Fetch the latest research papers and explore insights with AI.")
