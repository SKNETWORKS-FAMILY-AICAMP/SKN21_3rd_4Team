# workflow 전체 플로우 구현
# rerank_node 제거 버전

from langgraph.graph import StateGraph, START, END
from src.agent.nodes.analyst_node import analyst_node, web_search_node, check_relevance, no_data_node
from src.agent.nodes.search_node import search_node, build_context
from src.schema.state import AgentState

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("search_node", search_node)
    workflow.add_node("build_context", build_context)
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("no_data_node", no_data_node)

    workflow.add_edge(START, "search_node")
    workflow.add_edge("search_node", "build_context")  # rerank 없이 바로 build_context로
    workflow.add_conditional_edges(
        "build_context",
        check_relevance,
        {
            "no_data_node": "no_data_node",      # 0.3 이하
            "web_search_node": "web_search_node",  # 0.3 ~ 0.5
            "analyst_node": "analyst_node"          # 0.5 초과
        }
    )
    workflow.add_edge("no_data_node", END)  # analyst 거치지 않고 바로 종료
    workflow.add_edge("web_search_node", "analyst_node")
    workflow.add_edge("analyst_node", END)

    return workflow