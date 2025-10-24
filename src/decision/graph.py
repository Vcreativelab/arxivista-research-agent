# It creates the state graph for the decision-making pipeline,
# defines AgentState,
# and sets up the routing and tool execution functions.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from typing import List, TypedDict, Annotated
import operator

# Define the AgentState.
class AgentState(TypedDict):
    input: str
    messages: List[BaseMessage] # renamed
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

# Import tool functions from the decision module.
from src.decision.oracle import oracle
from src.tools.rag_search_filter import rag_search_filter
from src.tools.rag_search import rag_search
from src.tools.fetch_arxiv import fetch_arxiv
from src.tools.web_search import web_search
from src.tools.final_answer import final_answer

# Functions to execute oracle and tools.
# Log the oracle decisions for debugging
def run_oracle(state: dict) -> dict:
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]['name']
    print(f"🧭 Oracle chose: {tool_name}")
    tool_args = out.tool_calls[0]['args']
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log='TBD')
    return {'intermediate_steps': [action_out]}


# Add a recursion guard
def router(state: dict) -> str:
    steps = state.get('intermediate_steps', [])
    if len(steps) > 10:
        print("⚠️ Too many tool calls — forcing final_answer")
        return 'final_answer'
    if isinstance(steps, list):
        return steps[-1].tool
    return 'final_answer'

tool_str_to_func = {
    'rag_search_filter': rag_search_filter,
    'rag_search': rag_search,
    'fetch_arxiv': fetch_arxiv,
    'web_search': web_search,
    'final_answer': final_answer
}

def run_tool(state: dict) -> dict:
    """Executes the tool indicated by the current state's last action."""
    tool_name = state['intermediate_steps'][-1].tool
    tool_args = state['intermediate_steps'][-1].tool_input
    print(f"{tool_name} called with arguments: {tool_args}")
    # Call the function directly with keyword arguments.
    out = tool_str_to_func[tool_name](**tool_args)
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))
    return {'intermediate_steps': [action_out]}


# Construct the state graph.
graph = StateGraph(AgentState)
graph.add_node('oracle', run_oracle)
graph.add_node('rag_search_filter', run_tool)
graph.add_node('rag_search', run_tool)
graph.add_node('fetch_arxiv', run_tool)
graph.add_node('web_search', run_tool)
graph.add_node('final_answer', run_tool)

# Set entry point.
graph.set_entry_point('oracle')

# Add conditional edges using router.
graph.add_conditional_edges(source='oracle', path=router)

# Connect nodes: from every tool (except final_answer) back to oracle.
for tool in [rag_search_filter, rag_search, fetch_arxiv, web_search]:
    graph.add_edge(tool.__name__, 'oracle')
graph.add_edge('final_answer', END)

# Compile the graph.
runnable = graph.compile()

