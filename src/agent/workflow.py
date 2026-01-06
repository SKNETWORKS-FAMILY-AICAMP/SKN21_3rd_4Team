# workflow 전체 플로우 구현
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/multi-agent-collaboration.ipynb

from langgraph.graph import StateGraph, START, END
from src.agent.nodes import analyst_node, search_node, build_context
from src.schema.state import AgentState

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("search_node", search_node)
    workflow.add_node("build_context", build_context)
    workflow.add_node("analyst_node", analyst_node)

    workflow.add_edge(START, "search_node")
    workflow.add_edge("search_node", "build_context")
    workflow.add_edge("build_context", "analyst_node")
    workflow.add_edge("analyst_node", END)

    return workflow