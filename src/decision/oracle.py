# It sets up the Oracle using an LLM and binds the available tools.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolCall
from src.tools.rag_search_filter import rag_search_filter
from src.tools.rag_search import rag_search
from src.tools.fetch_arxiv import fetch_arxiv
from src.tools.web_search import web_search
from src.tools.final_answer import final_answer
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Define system prompt for the Oracle.
system_prompt = (
    "You are the oracle, an AI decision-maker. Given the user's query, "
    "decide which tool(s) to use from the list provided. Do not reuse a tool "
    "more than twice for the same query. Once you have gathered sufficient information, "
    "use the final_answer tool to produce the report."
)

# Create prompt template.
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', '{input}'),
    ('assistant', 'scratchpad: {scratchpad}'),
    MessagesPlaceholder(variable_name='messages'), # renamed
])

# Initialize the LLM.
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)

# List of available tools.
tools = [rag_search_filter, rag_search, fetch_arxiv, web_search, final_answer]

# Function to create the scratchpad from intermediate tool calls.
def create_scratchpad(intermediate_steps: list[ToolCall]) -> str:
    steps = []
    for action in intermediate_steps:
        if action.log != 'TBD':
            steps.append(f"Tool: {action.tool}, Input: {action.tool_input}\nOutput: {action.log}")
    return "\n---\n".join(steps)

# Construct the oracle pipeline.
oracle = (
    {
        'input': lambda x: x['input'],
        'chat_history': lambda x: x['messages'], # renamed
        'scratchpad': lambda x: create_scratchpad(x['intermediate_steps']),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice='any')
)
