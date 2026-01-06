from langchain_core.messages import HumanMessage
from langgraph.types import Command
from typing import Literal
from langgraph.graph import END
from src.agent.nodes.search_executor import SearchExecutor
from src.agent.nodes.search_agent import execute_dual_query_search
from src.agent.nodes.search_agent import build_search_config


def search_node(state: dict) -> dict:
    """
    LangGraph 노드용 함수
    state에서 query를 읽고, 검색 결과를 반환
    """
    
    executor = SearchExecutor()
    query = state['query']
    
    # 검색 설정 및 실행
    config = build_search_config(query)
    results, query_info = execute_dual_query_search(query, executor)
    
    # 결과 포맷팅
    top_k = config.get('top_k', 5)
    search_results = [
        {
            "content": r['content'],
            "score": round(r['score'], 4),
            "metadata": r['metadata']
        }
        for r in results[:top_k]
    ]
    
    return {
        'search_results': search_results,
        'context': executor.build_context(results)
    }