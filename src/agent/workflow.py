# workflow 전체 플로우 구현
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/multi-agent-collaboration.ipynb

from langgraph.graph import StateGraph, START, END
# from src.agent.nodes.search_agent import research_node
from src.agent.nodes.analyst_agent import analyst_node
from src.schema.state import AgentState

def build_graph():
    workflow = StateGraph(AgentState)
    # workflow.add_node("researcher", research_node)
    workflow.add_node("analyst_node", analyst_node)

    workflow.add_edge(START, "analyst_node")
    # workflow.add_edge("researcher", "chart_generator")
    workflow.add_edge("analyst_node", END)
    graph = workflow.compile()
    return graph