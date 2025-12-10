# src/decision/graph.py
# Decision graph for routing oracle → tools → oracle → final_answer.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from typing import List, TypedDict, Annotated, Dict
import operator

# ---------------- Agent State ----------------
class AgentState(TypedDict):
    input: str
    messages: List[BaseMessage]
    intermediate_steps: Annotated[List[AgentAction], operator.add]
    tool_usage: Dict[str, int]


# ---------------- Import oracle and tools ----------------
from src.decision.oracle import oracle
from src.tools.rag_search_filter import rag_search_filter
from src.tools.rag_search import rag_search
from src.tools.fetch_arxiv import fetch_arxiv
from src.tools.web_search import web_search
from src.tools.final_answer import final_answer


# ---------------- Run Oracle ----------------
def run_oracle(state: dict) -> dict:
    """
    Executes the LLM-based oracle which decides the next tool call.
    """
    out = oracle.invoke(state)

    tool_call = out.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"\n🧭 ORACLE → tool: {tool_name}, args: {tool_args}")

    # Update tool usage counter
    usage = state.get("tool_usage", {})
    usage[tool_name] = usage.get(tool_name, 0) + 1

    return {
        "intermediate_steps": [
            AgentAction(tool=tool_name, tool_input=tool_args, log="PENDING")
        ],
        "tool_usage": usage
    }


# ---------------- Router Logic ----------------
def router(state: dict) -> str:
    """
    Decides the next node based on last tool, recursion guard, and usage limits.
    """
    steps = state.get("intermediate_steps", [])
    usage = state.get("tool_usage", {})

    # recursion guard
    if len(steps) > 10:
        print("⚠️ Too many tool calls — forcing final_answer")
        return "final_answer"

    last_action = steps[-1]
    tool_name = last_action.tool

    # tool overuse guard
    if usage.get(tool_name, 0) > 2:
        print(f"⚠️ Tool {tool_name} used more than twice — switching to final_answer")
        return "final_answer"

    return tool_name


# ---------------- Tool Execution ----------------
tool_str_to_func = {
    "rag_search_filter": rag_search_filter,
    "rag_search": rag_search,
    "fetch_arxiv": fetch_arxiv,
    "web_search": web_search,
    "final_answer": final_answer
}


def run_tool(state: dict) -> dict:
    """
    Executes the chosen tool and returns its unified output into the state.
    """
    action = state["intermediate_steps"][-1]
    tool_name = action.tool
    tool_args = action.tool_input

    print(f"🔧 TOOL EXECUTION → {tool_name}({tool_args})")

    # run tool
    output = tool_str_to_func[tool_name](**tool_args)

    # update scratchpad entry
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=output  # Store full dict, not stringified
    )

    return {"intermediate_steps": [action_out]}


# ---------------- Build Graph ----------------
graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
graph.add_node("rag_search_filter", run_tool)
graph.add_node("rag_search", run_tool)
graph.add_node("fetch_arxiv", run_tool)
graph.add_node("web_search", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges(source="oracle", path=router)

# send all tools (except final_answer) back to oracle
for tool in ["rag_search_filter", "rag_search", "fetch_arxiv", "web_search"]:
    graph.add_edge(tool, "oracle")

graph.add_edge("final_answer", END)

runnable = graph.compile()
