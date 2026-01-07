from typing import TypedDict, List, Dict, Any
from typing import Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    LangGraph 워크플로우 상태 관리
    
    [데이터 흐름]
    START → search_node → build_context → check_relevance → analyst_node/web_search_node → END
    """
    
    # ─────────────────────────────────────────────────────────────
    # 0. 대화 기록 (MemorySaver와 함께 사용)
    # ─────────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]  # 대화 기록 자동 누적
    
    # ─────────────────────────────────────────────────────────────
    # 1. 입력 (Input) - main.py에서 초기화
    # ─────────────────────────────────────────────────────────────
    query: str  # 사용자의 원래 질문
    
    # ─────────────────────────────────────────────────────────────
    # 2. 검색 결과 (search_node → build_context)
    # ─────────────────────────────────────────────────────────────
    search_results: List[Dict[str, Any]]  # content, score, metadata 포함
    
    # ─────────────────────────────────────────────────────────────
    # 3. 컨텍스트 (build_context → analyst_node/web_search_node)
    # ─────────────────────────────────────────────────────────────
    context: str  # LLM이 참고할 포맷팅된 텍스트
    
    # ─────────────────────────────────────────────────────────────
    # 4. 분석 결과 (analyst_node → END) - 최종 출력
    # ─────────────────────────────────────────────────────────────
    analyst_results: Annotated[list, add_messages]
    
    # ─────────────────────────────────────────────────────────────
    # 5. 추천 질문 (no_data_node에서 설정)
    # ─────────────────────────────────────────────────────────────
    suggested_questions: List[str]
