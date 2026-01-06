"""
Search Agent - 듀얼 쿼리 검색 시스템

무엇을 하는 파일인가?
- 사용자 질문을 Qdrant(Vector DB)에서 검색해, 관련 문서 조각(top_k)을 가져오는 실행/테스트 스크립트입니다.
- Python 공식문서(RST)는 영어 본문이 대부분이라 한글 질문만으로는 유사도 점수가 낮게 나올 수 있어
  "원문(한글) + 번역(영어)"를 같이 검색해 recall을 올리는 전략(dual query)을 사용합니다.

1) 질문 언어 판별: `is_korean()`
2) 검색 설정 결정: `build_search_config(query)`
   - top_k, sources(lecture/python_doc_rst), search_method 등을 결정
3) 소스별 검색: `search_by_source(query, source, top_k)`
   - Qdrant에서 `metadata.source`로 필터링해 각각 검색 (lecture vs python_doc_rst)
4) (질문이 한글이면) 번역 검색 추가: `translate_to_english()`
   - 영어 키워드 쿼리로 한 번 더 소스별 검색
5) 결과 합치기 → 중복 제거 → 점수순 정렬 → 최종 top_k 반환

실행
- `python src/agent/nodes/search_agent.py`
"""
import sys
import os
import time
import re

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

# 로컬 실행 시 `src.` import가 깨지지 않게 프로젝트 루트를 path에 추가
sys.path.append(os.getcwd())

from src.agent.nodes.search_router import build_search_config
from src.agent.prompts import PROMPTS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.config import ConfigDB, ConfigAPI


# ============================================================
# 듀얼 쿼리 검색 함수
# ============================================================

def is_korean(text: str) -> bool:
    """한글 포함 여부 확인"""
    return bool(re.search(r'[가-힣]', text))


def create_translate_chain():
    """
    번역용 LangChain chain 생성
    
    Returns:
        Chain: prompt | llm | parser 형태의 chain
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(PROMPTS["TRANSLATE_PROMPT"])
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    return chain


def translate_to_english(query: str) -> str:
    """
    LLM으로 한글 → 영어 검색 쿼리 변환 (체인화 버전)
    
    Args:
        query: 한글 질문
        
    Returns:
        영어 검색 키워드
    """
    chain = create_translate_chain()
    return chain.invoke({"query": query}).strip()


def search_by_source(query: str, source: str, top_k: int) -> list:
    """특정 소스에서만 검색 (Qdrant 필터 사용)"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    client = QdrantClient(
            host=ConfigDB.HOST,
            port=ConfigDB.PORT
        )

    embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=ConfigAPI.OPENAI_API_KEY
        )
    
    query_vector = embeddings.embed_query(query)
    
    search_result = client.query_points(
        collection_name=ConfigDB.COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source)
                )
            ]
        ),
        limit=top_k
    )
    
    results = []
    for hit in search_result.points:
        results.append({
            "content": hit.payload.get('page_content', ''),
            "score": hit.score,
            "metadata": hit.payload.get('metadata', {})
        })
    return results


def execute_dual_query_search(query: str) -> tuple:
    """
    소스별 듀얼 쿼리 검색
    
    1. LLM이 top_k 결정 (basic=3, intermediate=5, advanced=7)
    2. lecture/python_doc 각각에서 top_k개씩 검색
    3. 합쳐서 유사도 순 정렬 → 최종 top_k 반환
    
    Returns:
        (results, query_info): 결과 리스트와 쿼리 정보
    """

    all_results = []
    query_info = {"original": query, "translated": None, "queries_used": []}
    
    # LLM이 top_k / sources 결정
    config = build_search_config(query)
    top_k = config.get('top_k', 5)
    sources = config.get("sources", ["lecture", "python_doc"])

    # Router는 python_doc을 주지만 Qdrant payload는 python_doc_rst를 쓰는 경우가 많음
    sources = ["python_doc_rst" if s == "python_doc" else s for s in sources]
    
    # 정책:
    # - lecture: (대부분 한국어 텍스트) 질문 원문으로만 검색
    # - python_doc_rst: (영어 문서) 한글 질문이면 번역(영어 키워드) 검색을 기본으로 하고,
    #                  결과가 약할 때만 한글 원문으로 fallback 검색
    PYDOC_FALLBACK_SCORE_THRESHOLD = 0.45

    # 1) lecture는 원문으로만 검색
    lecture_results = search_by_source(query, "lecture", top_k) if "lecture" in sources else []

    # 2) python_doc_rst 검색
    python_results = []
    if "python_doc_rst" in sources:
        if is_korean(query):
            # 2-1) 번역(영어 키워드) 검색이 기본
            english_query = translate_to_english(query)
            query_info["translated"] = english_query
            python_results_en = search_by_source(english_query, "python_doc_rst", top_k)
            for r in python_results_en:
                r["query_type"] = "translated"
            all_results.extend(python_results_en)
            query_info["queries_used"].append(f"번역(python_doc_rst): {english_query}")

            # 2-2) fallback: 번역 결과가 약하면 한글 원문으로도 한 번 더 검색
            best_score = python_results_en[0]["score"] if python_results_en else 0
            if (not python_results_en) or (best_score < PYDOC_FALLBACK_SCORE_THRESHOLD):
                python_results = search_by_source(query, "python_doc_rst", top_k)
        else:
            # 영어 질문이면 원문(영어) 그대로
            python_results = search_by_source(query, "python_doc_rst", top_k)
    else:
        python_results = []
    
    for r in lecture_results + python_results:
        r['query_type'] = 'original'
    all_results.extend(lecture_results + python_results)
    query_info["queries_used"].append(f"원본: {query}")
    
    # 3. 중복 제거
    seen = set()
    unique_results = []
    for r in all_results:
        content_key = r['content'].strip()[:100]
        if content_key not in seen:
            seen.add(content_key)
            unique_results.append(r)
    
    # 4. 유사도 순 정렬 후 top_k만 반환
    unique_results.sort(key=lambda x: x['score'], reverse=True)
    
    return unique_results[:top_k], query_info
