# pages/3_Ask_Research_Agent.py

import streamlit as st
from streamlit_lottie import st_lottie
import requests
from src.decision.graph import runnable
from src.tools.final_answer import format_final_answer

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Ask Arxivista Research Agent", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>🤖 Ask <span style='color:#6C63FF;'>Arxivista</span> Research Agent</h1>",
    unsafe_allow_html=True
)

# ---------------- Lottie Animation Loader ----------------
@st.cache_resource
def load_lottie_url(url: str):
    r = requests.get(url)
    try:
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.JSONDecodeError:
        return None

animation = load_lottie_url("https://lottie.host/dd0aede9-2d61-4564-8c3e-b947bc3cc41d/oHPVxVk0Hl.json")

# ---------------- Pinecone Index Loader ----------------
@st.cache_resource
def get_pinecone_index():
    from langchain_community.vectorstores import Pinecone
    from src.config import embeddings
    return Pinecone.from_existing_index(index_name="research-knowledge", embedding=embeddings)

# ---------------- Session State ----------------
if "query" not in st.session_state:
    st.session_state.query = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Adaptive Gradient Styling ----------------
st.markdown("""
    <style>
        /* Streamlit theme-aware styling */
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
            .stTextInput input {
                background-color: #FFFFFF;
                color: var(--text-color-light);
            }
        }

        @media (prefers-color-scheme: dark) {
            div.stButton > button:first-child {
                background: linear-gradient(90deg, var(--primary-color-dark), var(--secondary-color-dark));
                color: var(--text-color-dark);
            }
            .stTextInput input {
                background-color: #2B2B2B;
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

# ---------------- Query Input ----------------
st.markdown("### 💬 Ask a research question below")
query = st.text_input("Enter your query:", st.session_state.query, key="query_input")

col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    ask_pressed = st.button("Ask Agent")

# ---------------- Main Logic ----------------
if ask_pressed:
    if query.strip():
        user_query = query
        st.session_state.query = ""  # Reset box

        # Show animation
        animation_placeholder = st.empty()
        text_placeholder = st.empty()
        with animation_placeholder.container():
            st_lottie(animation, height=200, key="compiling")
            text_placeholder.write("📋 Compiling your personalized research report...")

        # Run pipeline
        output = runnable.invoke({'input': user_query, 'chat_history': st.session_state.chat_history})
        report = format_final_answer(output['intermediate_steps'][-1].tool_input)

        animation_placeholder.empty()
        text_placeholder.empty()

        # Display Output
        st.subheader("📜 Research Report")
        st.markdown(report)

        # Save chat
        st.session_state.chat_history.append({"query": user_query, "response": report})
    else:
        st.warning("⚠️ Please enter a valid research question.")

# ---------------- Chat History ----------------
with st.expander("🕓 Show Chat History", expanded=False):
    if st.session_state.chat_history:
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{i+1}:** {entry['query']}")
            st.markdown(f"**A{i+1}:** {entry['response']}")
            st.markdown("---")
    else:
        st.info("No previous queries yet.")
