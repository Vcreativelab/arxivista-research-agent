import streamlit as st
from streamlit_lottie import st_lottie
import requests
from src.decision.graph import runnable
from src.tools.final_answer import format_final_answer

st.set_page_config(page_title="Ask Research Agent", layout="wide")
st.title("🤖 Ask Research Agent")

# ---------------- Lottie animation loader ----------------
@st.cache_resource
def load_lottie_url(url: str):
    r = requests.get(url)
    try:
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.JSONDecodeError:
        return None

animation = load_lottie_url("https://lottie.host/dd0aede9-2d61-4564-8c3e-b947bc3cc41d/oHPVxVk0Hl.json")

# ---------------- Pinecone index loader ----------------
@st.cache_resource
def get_pinecone_index():
    from langchain_community.vectorstores import Pinecone
    from src.config import embeddings
    return Pinecone.from_existing_index(index_name="research-knowledge", embedding=embeddings)

# ---------------- Session state initialization ----------------
if "query" not in st.session_state:
    st.session_state.query = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- User input ----------------
query = st.text_input("Enter your query:", st.session_state.query, key="query_input")

if st.button("Ask Agent"):
    if st.session_state.query.strip():
        user_query = st.session_state.query
        st.session_state.query = ""  # Clear input box immediately

        # Show compiling animation
        animation_placeholder = st.empty()
        text_placeholder = st.empty()
        with animation_placeholder.container():
            st_lottie(animation, height=200, key="compiling")
            text_placeholder.write("📋 Compiling the report...")

        # Run pipeline
        output = runnable.invoke({'input': user_query, 'chat_history': []})
        report = format_final_answer(output['intermediate_steps'][-1].tool_input)

        animation_placeholder.empty()
        text_placeholder.empty()

        # Display final output
        st.subheader("📜 Research Report")
        st.markdown(report)

        # Save query and response in chat history
        st.session_state.chat_history.append({
            "query": user_query,
            "response": report
        })
    else:
        st.warning("Please enter a valid query.")

# ---------------- Collapsible chat history ----------------
with st.expander("📜 Show Chat History", expanded=False):
    if st.session_state.chat_history:
        for i, entry in enumerate(st.session_state.chat_history[::-1]):  # latest first
            st.markdown(f"**Q{i+1}:** {entry['query']}")
            st.markdown(f"**A{i+1}:** {entry['response']}")
            st.markdown("---")
    else:
        st.info("No previous queries yet.")
