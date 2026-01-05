from typing import TypedDict, List, Dict, Any, Optional

class SearchConfig(TypedDict):
    """
    Search Router(Role A) -> Search Executor(Role B) 전달용 설정
    """
    sources: List[str]          # 검색 대상 소스 ["lecture", "python_doc", "web"]
    top_k: int                  # 반환할 문서 개수
    search_method: str          # [추가] 'similarity' 또는 'mmr'
    filters: Optional[Dict[str, Any]]  # (선택) 메타데이터 필터 조건