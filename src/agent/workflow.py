# workflow 전체 플로우 구현
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/multi-agent-collaboration.ipynb

from langgraph.graph import StateGraph, START, END
from src.agent.nodes.analyst_node import analyst_node, web_search_node, check_relevance
from src.agent.nodes.search_node import search_node, build_context
from src.schema.state import AgentState

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("search_node", search_node)
    workflow.add_node("build_context", build_context)
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("web_search_node", web_search_node)

    workflow.add_edge(START, "search_node")
    workflow.add_edge("search_node", "build_context")
    workflow.add_conditional_edges(
                                    "build_context",
                                    check_relevance,
                                    {
                                        "web_search_node": "web_search_node",
                                        "analyst_node": "analyst_node" # 바로 분석가로
                                    }
                                )
    workflow.add_edge("web_search_node", "analyst_node")
    workflow.add_edge("analyst_node", END)

    return workflow