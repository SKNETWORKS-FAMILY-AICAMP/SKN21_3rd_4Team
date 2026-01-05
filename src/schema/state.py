from typing import TypedDict, List, Dict, Any, Optional
from src.schema.search import SearchConfig
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Search Agent 전체 상태 관리
    Role A(Router)와 Role B(Executor) 간의 데이터 전달 통로
    """
    
    # 1. 입력 (Input)
    question: str              # 사용자의 원래 질문
    
    # [추가] 검색 키워드 (Role A가 질문을 구체화했을 경우 사용, 없으면 question 사용)
    search_query: str
    
    # 2. Role A (Router) -> Role B (Executor)
    # 정의된 SearchConfig 타입 사용 (src/schema/search.py)
    search_config: SearchConfig
    
    # 3. Role B (Executor) -> Analysis Agent
    # 실행된 원본 검색 결과 리스트 (메타데이터 포함)
    search_results: List[Dict[str, Any]]

    analyst_results: Annotated[list, add_messages]
    
    # LLM이 답변 생성에 참고할 최종 텍스트 문자열
    # (제목, 내용, 출처 등이 포맷팅된 형태)
    context: str
    
    # 4. 최종 답변 (Output)
    answer: str

