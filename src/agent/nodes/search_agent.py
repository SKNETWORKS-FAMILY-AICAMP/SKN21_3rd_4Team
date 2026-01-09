"""
Search Agent - 듀얼 쿼리 검색 시스템

이 모듈의 역할:
- 사용자 질문을 Qdrant(Vector DB)에서 검색해, 관련 문서 조각(top_k)을 가져옵니다
- Python 공식문서(RST)는 영어 본문이 대부분이라 한글 질문만으로는 유사도 점수가 낮게 나올 수 있어
  "원문(한글) + 번역(영어)"를 같이 검색해 recall을 올리는 전략(dual query)을 사용합니다
- 검색 흐름:
  1) 질문 언어 판별: `is_korean()`
  2) 검색 설정 결정: `build_search_config(query)` - top_k, sources 등 결정
  3) 소스별 검색: `search_by_source(query, source, top_k)` - lecture/python_doc 필터링
  4) (한글이면) 번역 검색 추가: `translate_to_english()` - 영어 키워드로 재검색
  5) 결과 합치기 → 중복 제거 → 점수순 정렬 → 최종 top_k 반환

핵심 개념: 하이브리드 검색
- 벡터 검색 + 키워드 매칭 + BM25로 검색 품질 향상
- 소스별 필터링으로 lecture/python_doc 구분 검색
- 한글 질문은 번역하여 영어 문서 검색 recall 향상
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
from src.utils.config import ConfigDB, ConfigAPI
from src.utils.search_utils import (
    is_korean,
    translate_to_english,
    calculate_keyword_score,
    calculate_bm25_score
)
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional


def search_by_source(query: str, source: str, top_k: int, keywords: Optional[List[str]] = None) -> list:
    """
    특정 소스에서만 검색 (Qdrant 필터 사용)
    하이브리드 검색(벡터 + 키워드 매칭 + BM25)을 사용합니다.
    
    Args:
        query: 검색 쿼리
        source: 소스 필터 ("lecture" 또는 "python_doc")
        top_k: 반환할 결과 수
        keywords: 명시적으로 지정된 키워드 리스트 (None이면 query에서 단순 추출)
    """
    client = QdrantClient(
        host=ConfigDB.HOST,
        port=ConfigDB.PORT
    )

    embeddings = OpenAIEmbeddings(
        model=ConfigDB.EMBEDDING_MODEL,
        api_key=ConfigAPI.OPENAI_API_KEY
    )
    
    query_words = query.split()
    is_single_word = len(query_words) == 1
    candidate_k = min(top_k * 6, 30) if is_single_word else min(top_k * 4, 20)
    query_vector = embeddings.embed_query(query)
    
    vector_result = client.query_points(
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
        limit=candidate_k
    )
    
    if keywords:
        query_keywords = [k.strip() for k in keywords if k.strip()]
    else:
        query_cleaned = query.replace(',', ' ').replace(';', ' ').replace(':', ' ')
        query_keywords = [kw.strip() for kw in query_cleaned.split() if len(kw.strip()) > 2]
        if not query_keywords and len(query_cleaned.strip()) > 2:
            query_keywords = [query_cleaned.strip()]
    
    candidates = []
    bm25_scores = []
    seen_ids = set()
    
    for hit in vector_result.points:
        content = hit.payload.get('page_content', '')
        vector_score = hit.score
        keyword_score = calculate_keyword_score(query_keywords, content)
        bm25_raw = calculate_bm25_score(query_keywords, content)
        bm25_scores.append(bm25_raw)
        
        doc_id = hit.id
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            candidates.append({
                "content": content,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "bm25_raw": bm25_raw,
                "metadata": hit.payload.get('metadata', {})
            })
    
    if is_single_word and len(candidates) < top_k:
        all_points = client.scroll(
            collection_name=ConfigDB.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source)
                    )
                ]
            ),
            limit=1000,
            with_payload=True
        )
        
        for point in all_points[0]:
            if point.id in seen_ids:
                continue
            
            content = point.payload.get('page_content', '')
            keyword_score = calculate_keyword_score(query_keywords, content)
            bm25_raw = calculate_bm25_score(query_keywords, content)
            
            if keyword_score > 0 or bm25_raw > 0:
                bm25_scores.append(bm25_raw)
                seen_ids.add(point.id)
                candidates.append({
                    "content": content,
                    "vector_score": 0.1,
                    "keyword_score": keyword_score,
                    "bm25_raw": bm25_raw,
                    "metadata": point.payload.get('metadata', {})
                })
                
                if len(candidates) >= top_k * 3:
                    break
    
    # BM25 점수 정규화 (0~1 범위로)
    if bm25_scores and max(bm25_scores) > 0:
        max_bm25 = max(bm25_scores)
        for i, candidate in enumerate(candidates):
            candidate['bm25_score'] = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0.0
    else:
        for candidate in candidates:
            candidate['bm25_score'] = 0.0
    
    # 하이브리드 점수 계산 (벡터 + 키워드 + BM25)
    # 단일 단어 쿼리일 때는 키워드/BM25 가중치를 높여서 정확한 매칭 강조
    if is_single_word:
        # 단일 단어: 키워드 매칭과 BM25가 더 중요 (벡터는 보조)
        vector_weight = 0.4
        keyword_weight = 0.3
        bm25_weight = 0.3
    else:
        # 일반 쿼리: vector 0.6, keyword 0.2, bm25 0.2 (vector_search.py와 동일)
        vector_weight = 0.6
        keyword_weight = 0.2
        bm25_weight = 0.2
    
    for candidate in candidates:
        hybrid_score = (
            candidate['vector_score'] * vector_weight +
            candidate['keyword_score'] * keyword_weight +
            candidate['bm25_score'] * bm25_weight
        )
        candidate['score'] = hybrid_score
    
    # 단일 단어 쿼리일 때 키워드/BM25 매칭이 있는 문서는 최소 점수 보장
    if is_single_word:
        for candidate in candidates:
            if candidate['keyword_score'] > 0 or candidate['bm25_score'] > 0:
                # 키워드나 BM25 매칭이 있으면 최소 점수 0.3 보장
                candidate['score'] = max(candidate['score'], 0.3)
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_k]


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
    analysis = config.get('_analysis', {})
    topic_keywords = analysis.get('topic_keywords', [])
    
    # 정책:
    # - lecture: (대부분 한국어 텍스트) 질문 원문으로만 검색
    # - python_doc: (영어 문서) 한글 질문이면 번역(영어 키워드) 검색을 기본으로 하고,
    #                  결과가 약할 때만 한글 원문으로 fallback 검색
    PYDOC_FALLBACK_SCORE_THRESHOLD = 0.45

    # 1) lecture는 원문으로만 검색
    lecture_results = search_by_source(query, "lecture", top_k, keywords=topic_keywords) if "lecture" in sources else []

    # 2) python_doc 검색
    python_results = []
    if "python_doc" in sources:
        if is_korean(query):
            # 2-1) 번역(영어 키워드) 검색이 기본
            english_query = translate_to_english(query)
            query_info["translated"] = english_query
            # python_doc(영어 번역 검색)은 영어 키워드이므로 topic_keywords(한글) 대신 쿼리에서 자동 추출하도록 None 전달
            python_results_en = search_by_source(english_query, "python_doc", top_k)
            for r in python_results_en:
                r["query_type"] = "translated"
            all_results.extend(python_results_en)
            query_info["queries_used"].append(f"번역(python_doc): {english_query}")

            # 2-2) fallback: 번역 결과가 약하면 한글 원문으로도 한 번 더 검색
            best_score = python_results_en[0]["score"] if python_results_en else 0
            if (not python_results_en) or (best_score < PYDOC_FALLBACK_SCORE_THRESHOLD):
                python_results = search_by_source(query, "python_doc", top_k, keywords=topic_keywords)
        else:
            # 영어 질문이면 원문(영어) 그대로
            python_results = search_by_source(query, "python_doc", top_k)
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
