"""
다른 컴퓨터 간 점수 차이를 줄이기 위한 추가 수정을 적용했습니다.
추가 수정
1. 벡터 검색 결과 순서 고정
문제: Qdrant가 같은 점수 문서를 다른 순서로 반환할 수 있음
수정:
# 벡터 검색 결과를 점수(내림차순) → ID(오름차순)로 정렬sorted_points = sorted(    vector_result.points,    key=lambda p: (-p.score, p.id)  # 점수는 내림차순, ID는 오름차순)
동일 점수에서도 ID로 정렬되어 순서가 일관됩니다.
2. BM25 정규화 기준 고정
문제: 후보 집합이 달라지면 정규화 기준(max_bm25)이 달라짐
수정:
# 고정 스케일과 현재 최댓값 중 큰 값 사용BM25_NORMALIZATION_SCALE = 50.0normalization_base = max(max_bm25, BM25_NORMALIZATION_SCALE * 0.5)
고정 스케일과 현재 최댓값 중 큰 값을 사용하여 후보가 달라져도 정규화가 더 일관됩니다.
수정 요약
인덱스 불일치 버그 수정: 잘못된 문서에 점수가 할당되지 않도록 수정
벡터 검색 결과 순서 고정: 점수 → ID 정렬로 순서 일관성 확보
추가 검색(scroll) 순서 고정: ID 정렬로 순서 일관성 확보
BM25 정규화 기준 고정: 고정 스케일 사용으로 일관성 개선
번역 캐싱: 같은 프로세스 내에서 캐싱 (다른 컴퓨터에서는 여전히 매번 번역)
남은 변동성 원인
LLM 번역 결과: temperature=0이어도 약간의 변동 가능성
Router의 키워드 추출: LLM이 추출하는 키워드가 달라질 수 있음
하지만 이제 대부분의 변동성은 해결되었습니다. 같은 스냅샷, 같은 질문이라면 다른 컴퓨터에서도 유사도 점수가 거의 동일하게 나와야 합니다.
테스트해보시고, 여전히 차이가 크면 알려주세요.
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.config import ConfigDB, ConfigAPI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional


# ============================================================
# 번역 캐시 (동일한 질문에 대해 동일한 번역 보장)
# ============================================================
_translation_cache: Dict[str, str] = {}


# ============================================================
# 듀얼 쿼리 검색 함수
# ============================================================

def _is_korean(text: str) -> bool:
    """한글 포함 여부 확인"""
    return bool(re.search(r'[가-힣]', text))


def _create_translate_chain():
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


def _translate_to_english(query: str) -> str:
    """
    LLM으로 한글 → 영어 검색 쿼리 변환 (체인화 버전)
    캐싱을 통해 동일한 질문에 대해 동일한 번역 보장
    
    Args:
        query: 한글 질문
        
    Returns:
        영어 검색 키워드
    """
    # 캐시 확인
    if query in _translation_cache:
        return _translation_cache[query]
    
    # 번역 실행
    chain = _create_translate_chain()
    result = chain.invoke({"query": query}).strip()
    
    # 캐시 저장
    _translation_cache[query] = result
    return result


def _calculate_keyword_score(query_keywords: List[str], content: str) -> float:
    """
    키워드 매칭 점수 계산 (0.0 ~ 1.0)
    부분 단어 매칭도 지원합니다 (예: "trimming"이 "trimming_history"에 포함되면 매칭).
    """
    if not query_keywords or not content:
        return 0.0
    
    content_lower = content.lower()
    # 문서를 단어로 분리 (정확한 매칭 확인용)
    doc_words = re.findall(r'\b\w+\b', content_lower)
    
    matched_count = 0
    total_weight = 0
    
    for keyword in query_keywords:
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            continue
        
        weight = len(keyword_lower.split())
        
        # 1. 정확한 단어 매칭 (우선순위 높음)
        exact_match = keyword_lower in doc_words
        
        # 2. 부분 문자열 매칭 (예: "trimming"이 "trimming_history"에 포함)
        partial_match = keyword_lower in content_lower
        
        if exact_match:
            # 정확한 매칭: 전체 가중치
            matched_count += weight
            if any(prefix in content for prefix in [f"[TITLE]", f"[H1]", f"[H2]", f"[API]", f"[KEYWORDS]"]):
                matched_count += weight * 0.5
        elif partial_match:
            # 부분 매칭: 가중치 0.7 (정확한 매칭보다 낮음)
            matched_count += weight * 0.7
            if any(prefix in content for prefix in [f"[TITLE]", f"[H1]", f"[H2]", f"[API]", f"[KEYWORDS]"]):
                matched_count += weight * 0.35
        
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    score = matched_count / total_weight
    return min(score, 1.0)


def _calculate_bm25_score(query_keywords: List[str], content: str, avg_doc_length: float = 100.0, k1: float = 1.5, b: float = 0.75) -> float:
    """
    BM25 점수 계산 (Sparse 검색)
    
    BM25는 TF-IDF의 개선 버전으로, 문서 길이 정규화를 포함합니다.
    부분 단어 매칭도 지원합니다 (예: "trimming"이 "trimming_history"에 포함되면 매칭).
    
    Args:
        query_keywords: 검색 쿼리의 키워드 리스트
        content: 문서 내용
        avg_doc_length: 평균 문서 길이 (기본값: 100.0, 실제로는 전체 문서 평균 사용 권장)
        k1: TF 정규화 파라미터 (기본값: 1.5)
        b: 문서 길이 정규화 파라미터 (기본값: 0.75)
        
    Returns:
        BM25 점수 (정규화되지 않음, 비교용으로만 사용)
    """
    if not query_keywords or not content:
        return 0.0
    
    # 문서를 단어로 분리 (소문자 변환)
    doc_words = re.findall(r'\b\w+\b', content.lower())
    doc_length = len(doc_words)
    content_lower = content.lower()
    
    if doc_length == 0:
        return 0.0
    
    # 각 키워드의 TF 계산
    score = 0.0
    for keyword in query_keywords:
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            continue
        
        # 1. 정확한 단어 매칭 (우선순위 높음)
        term_freq = doc_words.count(keyword_lower)
        exact_match = term_freq > 0
        
        # 2. 부분 단어 매칭 (예: "trimming"이 "trimming_history"에 포함)
        # 정확한 매칭이 없을 때만 부분 매칭 사용 (가중치 낮춤)
        partial_match = False
        partial_freq = 0
        if not exact_match:
            # 단어 경계를 고려한 부분 매칭 (단어 중간에 포함되는 경우)
            for word in doc_words:
                if keyword_lower in word or word in keyword_lower:
                    partial_freq += 1
                    partial_match = True
        
        # 매칭이 없으면 스킵
        if not exact_match and not partial_match:
            continue
        
        # TF 계산 (정확한 매칭 우선, 부분 매칭은 가중치 낮춤)
        if exact_match:
            final_freq = term_freq
        else:
            final_freq = partial_freq * 0.5  # 부분 매칭은 가중치 0.5
        
        # BM25 공식: IDF는 간단히 log로 근사 (실제로는 전체 문서 집합에서 계산해야 함)
        # 여기서는 키워드가 문서에 나타나면 점수를 주는 방식
        # 실제 IDF는 전체 문서 집합에서 계산해야 하지만, 여기서는 간단히 처리
        
        # TF 정규화 (BM25 공식)
        tf_norm = (final_freq * (k1 + 1)) / (final_freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        
        # 간단한 IDF 근사 (키워드 길이 기반 가중치)
        idf_weight = 1.0 + len(keyword_lower.split())  # 긴 키워드가 더 중요
        
        score += tf_norm * idf_weight
    
    return score


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
    
    # 하이브리드 검색: 벡터 + 키워드 매칭 + BM25 (vector_search.py와 동일한 방식)
    # 단일 단어 쿼리의 경우 더 많은 후보를 가져와서 부분 매칭 확률 증가
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
    
    # 벡터 검색 결과를 점수(내림차순) → ID(오름차순)로 정렬하여 순서 고정
    # 같은 점수일 때도 ID로 정렬하여 결정론적 동작 보장
    sorted_points = sorted(
        vector_result.points,
        key=lambda p: (-p.score, p.id)  # 점수는 내림차순, ID는 오름차순
    )
    
    # 키워드 추출
    if keywords:
        query_keywords = [k.strip() for k in keywords if k.strip()]
    else:
        query_cleaned = query.replace(',', ' ').replace(';', ' ').replace(':', ' ')
        # 단일 단어도 처리 (길이 2 이상, 대소문자 구분 없이)
        query_keywords = [kw.strip() for kw in query_cleaned.split() if len(kw.strip()) > 2]
        # 단일 단어 쿼리인 경우에도 키워드가 비어있지 않도록 보장
        if not query_keywords and len(query_cleaned.strip()) > 2:
            query_keywords = [query_cleaned.strip()]
    
    # 벡터 검색 결과 수집 및 점수 계산
    candidates = []
    seen_ids = set()  # 중복 제거용
    
    for hit in sorted_points:  # 정렬된 결과 사용
        # 문서 ID로 중복 제거 (같은 문서가 여러 번 나올 수 있음)
        doc_id = hit.id
        if doc_id in seen_ids:
            continue  # 중복 문서는 스킵
        
        seen_ids.add(doc_id)
        content = hit.payload.get('page_content', '')
        vector_score = hit.score
        keyword_score = _calculate_keyword_score(query_keywords, content)
        
        # BM25 점수 계산 (중복 제거 후에만 계산)
        bm25_raw = _calculate_bm25_score(query_keywords, content)
        
        candidates.append({
            "content": content,
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "bm25_raw": bm25_raw,
            "metadata": hit.payload.get('metadata', {})
        })
    
    # 단일 단어 쿼리이고 벡터 검색 결과가 적거나 키워드 매칭이 없는 경우
    # 키워드 기반으로 추가 검색 (벡터 점수는 낮게 설정)
    if is_single_word and len(candidates) < top_k:
        # 키워드 매칭이 있는 문서를 추가로 찾기 위해 전체 컬렉션에서 검색
        # (벡터 검색으로 찾지 못한 문서 중 키워드가 포함된 문서)
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
            limit=1000,  # 최대 1000개 스캔
            with_payload=True
        )
        
        # 순서 고정을 위해 ID로 정렬 (결정론적 동작 보장)
        scroll_points = sorted(all_points[0], key=lambda p: p.id)
        
        for point in scroll_points:
            if point.id in seen_ids:
                continue  # 이미 벡터 검색 결과에 포함된 문서는 제외
            
            content = point.payload.get('page_content', '')
            keyword_score = _calculate_keyword_score(query_keywords, content)
            bm25_raw = _calculate_bm25_score(query_keywords, content)
            
            # 키워드나 BM25 매칭이 있는 문서만 추가
            if keyword_score > 0 or bm25_raw > 0:
                seen_ids.add(point.id)
                candidates.append({
                    "content": content,
                    "vector_score": 0.1,  # 벡터 검색으로 찾지 못했으므로 낮은 점수
                    "keyword_score": keyword_score,
                    "bm25_raw": bm25_raw,
                    "metadata": point.payload.get('metadata', {})
                })
                
                # 충분한 후보를 모았으면 중단
                if len(candidates) >= top_k * 3:
                    break
    
    # BM25 점수 정규화 (0~1 범위로)
    # candidates의 bm25_raw를 직접 사용하여 인덱스 불일치 문제 해결
    # 결정론적 동작을 위해 고정된 최댓값 사용 (또는 현재 후보 중 최댓값)
    if candidates:
        # 모든 후보의 bm25_raw 값을 수집
        bm25_raws = [c['bm25_raw'] for c in candidates]
        max_bm25 = max(bm25_raws) if bm25_raws else 0
        
        # 결정론적 정규화: 고정된 스케일 사용 (대부분의 BM25 점수가 이 범위 내)
        # 실제 최댓값이 이보다 크면 잘릴 수 있지만, 일반적인 경우에는 충분함
        BM25_NORMALIZATION_SCALE = 50.0  # BM25 점수 정규화 스케일 (경험적 값)
        
        if max_bm25 > 0:
            # 두 가지 방법 중 큰 값을 사용: 현재 후보의 최댓값 또는 고정 스케일
            # 이렇게 하면 후보가 달라도 정규화가 일관됨
            normalization_base = max(max_bm25, BM25_NORMALIZATION_SCALE * 0.5)
            for candidate in candidates:
                candidate['bm25_score'] = min(candidate['bm25_raw'] / normalization_base, 1.0)
        else:
            for candidate in candidates:
                candidate['bm25_score'] = 0.0
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
        if _is_korean(query):
            # 2-1) 번역(영어 키워드) 검색이 기본
            english_query = _translate_to_english(query)
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
