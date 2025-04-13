import streamlit as st
from streamlit_lottie import st_lottie
import requests
from src.decision.graph import runnable
from src.tools.final_answer import format_final_answer

st.set_page_config(page_title="Ask Research Agent", layout="wide")
st.title("🤖 Ask Research Agent")


# Cache Lottie animations, so they don't reload every time
@st.cache_resource
def load_lottie_url(url: str):
    """Fetches Lottie animation JSON from an online source and caches it."""
    r = requests.get(url)
    try:
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.JSONDecodeError:
        return None


# Load cached animations
animation = load_lottie_url("https://lottie.host/dd0aede9-2d61-4564-8c3e-b947bc3cc41d/oHPVxVk0Hl.json")

# Cache the Pinecone connection (prevents reloading on every page switch)
@st.cache_resource
def get_pinecone_index():
    """Initializes Pinecone index only once."""
    from src.data.embeddings import PineconeVectorStore
    from src.config import embeddings
    return PineconeVectorStore(index_name="research-knowledge", embedding=embeddings)


# User Input
query = st.text_input("Enter your query:", "")

if query:
    # Create a placeholder for animations
    animation_placeholder = st.empty()
    text_placeholder = st.empty()

    with animation_placeholder.container():
        st_lottie(animation, height=200, key="compiling")
        text_placeholder.write("📋 Compiling the report...")

    # Run the pipeline
    output = runnable.invoke({'input': query, 'chat_history': []})

    # Generate Report
    report = format_final_answer(output['intermediate_steps'][-1].tool_input)

    # Remove finalizing animation before showing final output
    animation_placeholder.empty()
    text_placeholder.empty()

    # Display Final Output
    st.subheader("📜 Research Report")
    st.markdown(report)
