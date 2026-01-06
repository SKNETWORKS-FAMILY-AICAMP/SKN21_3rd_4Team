from src.schema.state import AgentState
from src.agent.nodes.search_agent import execute_dual_query_search


def search_node(state: AgentState):
    """
    LangGraph 노드용 함수
    state에서 query를 읽고, 검색 결과를 반환
    """
    
    query = state['query']
    
    # 검색 설정 및 실행
    results, query_info = execute_dual_query_search(query)
    
    # 결과 포맷팅
    search_results = [
        {
            "content": r['content'],
            "score": round(r['score'], 4),
            "metadata": r['metadata']
        }
        for r in results
    ]
    
    return {
        'search_results': search_results,
    }
    

def build_context(state: AgentState):
    """
    보고서 작성: LLM이 읽기 좋게 문장으로 정리합니다.
    """

    results = state['search_results']
    
    if not results:
        return "검색된 관련 자료가 없습니다."
    
    context_parts = []
    
    for i, res in enumerate(results, 1): # 번호는 1번부터
        source = res['metadata'].get('source', 'Unknown')
        score = round(res['score'], 2)
        content = res['content'].strip()
        
        part = f"[{i}] 출처: {source} (유사도: {score})\n{content}"
        context_parts.append(part)
        
    context = "\n\n---\n\n".join(context_parts)

    return {"context": context}
