# src/agent/graph.py

from langgraph.graph import StateGraph, END
from src.agent.nodes.analyst_node import build_analyst_node

def build_graph(llm):
    graph = StateGraph(dict)

    analyst_node = build_analyst_node(llm)

    graph.add_node("analyst", analyst_node)

    graph.set_entry_point("analyst")
    graph.add_edge("analyst", END)

    return graph.compile()
