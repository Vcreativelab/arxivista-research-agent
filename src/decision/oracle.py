# src/decision/oracle.py
# LLM oracle that selects the next tool.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolCall, AIMessage, HumanMessage
from dotenv import load_dotenv

from src.tools.rag_search_filter import rag_search_filter
from src.tools.rag_search import rag_search
from src.tools.fetch_arxiv import fetch_arxiv
from src.tools.web_search import web_search
from src.tools.final_answer import final_answer

load_dotenv()


# ---------------- System Prompt ----------------
system_prompt = (
    "You are the Oracle — an AI controller deciding which tool to call next.\n"
    "Rules:\n"
    "- Choose the best tool based on the user's query and prior tool results.\n"
    "- Do NOT use any tool more than twice.\n"
    "- If you already have enough relevant information, call 'final_answer'.\n"
    "- If the question is conceptual or opinion-based, call 'final_answer'.\n"
    "- Avoid loops; if in doubt, call 'final_answer'.\n"
)


# ---------------- Scratchpad Formatting ----------------
def create_scratchpad(intermediate_steps):
    """
    Convert tool call logs into clean readable JSON blocks.
    """
    lines = []
    for action in intermediate_steps:
        if isinstance(action.log, dict):  # our new unified schema
            tool_block = json.dumps(action.log, indent=2)
        else:
            tool_block = str(action.log)

        lines.append(
            f"TOOL USED: {action.tool}\n"
            f"INPUT: {json.dumps(action.tool_input, indent=2)}\n"
            f"OUTPUT:\n{tool_block}"
        )

    return "\n\n---\n\n".join(lines)


# ---------------- Oracle Pipeline ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="messages"),
    ("assistant", "Previous tool calls:\n{scratchpad}")
])

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

tools = [rag_search_filter, rag_search, fetch_arxiv, web_search, final_answer]

oracle = (
    {
        "input": lambda s: s["input"],
        "messages": lambda s: s["messages"],
        "scratchpad": lambda s: create_scratchpad(s["intermediate_steps"]),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)
