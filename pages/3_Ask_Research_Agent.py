# pages/3_Ask_Research_Agent.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from streamlit_lottie import st_lottie
import requests

from langchain_core.messages import HumanMessage, AIMessage

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

animation = load_lottie_url(
    "https://lottie.host/dd0aede9-2d61-4564-8c3e-b947bc3cc41d/oHPVxVk0Hl.json"
)


# ---------------- Session State ----------------
if "query" not in st.session_state:
    st.session_state.query = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# Sidebar Debug Toggle
st.sidebar.markdown("## 🛠 Debug Panel")
debug_on = st.sidebar.checkbox("Enable Debug Mode", value=False)


# ---------------- Styling ----------------
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
        st.session_state.query = ""

        # Prepare LangChain-compatible chat history
        messages = []
        for item in st.session_state.chat_history:
            messages.append(HumanMessage(content=item["query"]))
            messages.append(AIMessage(content=item["response"]))

        # Show animation
        animation_placeholder = st.empty()
        text_placeholder = st.empty()
        with animation_placeholder.container():
            st_lottie(animation, height=200, key="compiling")
            text_placeholder.write("📋 Compiling your personalized research report...")

        # Call the LangGraph pipeline
        output = runnable.invoke({
            "input": user_query,
            "messages": messages,
            "intermediate_steps": [],
            "tool_usage": {}
        })

        # Extract final tool output (NOT tool_input)
        final_action = output["intermediate_steps"][-1]
        final_output = final_action.log  # dict from final_answer()

        report = format_final_answer(final_output)

        animation_placeholder.empty()
        text_placeholder.empty()

        # Display report
        st.subheader("📜 Research Report")
        st.markdown(report)

        # Save conversation
        st.session_state.chat_history.append({
            "query": user_query,
            "response": report
        })

        # Save debug info
        st.session_state.debug_logs.append({
            "user_query": user_query,
            "oracle_tool": final_action.tool,
            "args": final_action.tool_input,
            "output": final_action.log,
            "all_steps": output["intermediate_steps"]
        })

    else:
        st.warning("⚠️ Please enter a valid research question.")


# ---------------- Debug Panel ----------------
if debug_on:
    st.sidebar.markdown("### 🧩 Debug Information")

    if st.session_state.debug_logs:
        last_debug = st.session_state.debug_logs[-1]

        st.sidebar.markdown("**Last Tool Used:**")
        st.sidebar.code(last_debug["oracle_tool"])

        st.sidebar.markdown("**Tool Arguments:**")
        st.sidebar.json(last_debug["args"])

        st.sidebar.markdown("**Tool Output (raw):**")
        st.sidebar.json(last_debug["output"])

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Intermediate Steps")

        for i, step in enumerate(last_debug["all_steps"]):
            st.sidebar.markdown(f"#### Step {i+1}: {step.tool}")
            st.sidebar.json({
                "input": step.tool_input,
                "output": step.log
            })
            st.sidebar.markdown("---")
    else:
        st.sidebar.info("No debug data yet.")


# ---------------- Chat History ----------------
with st.expander("🕓 Show Chat History", expanded=False):
    if st.session_state.chat_history:
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{i+1}:** {entry['query']}")
            st.markdown(f"**A{i+1}:** {entry['response']}")
            st.markdown("---")
    else:
        st.info("No previous queries yet.")
