"""
검색 유틸리티 함수 모음

search_agent.py와 vector_search.py에서 공통으로 사용되는 함수들을 모아놓은 모듈입니다.
- 한글 체크
- 번역 체인
- 키워드 점수 계산
- BM25 점수 계산
"""
import re
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agent.prompts import PROMPTS
from src.utils.config import ConfigLLM


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
    
    llm = ChatOpenAI(model=ConfigLLM.OPENAI_MODEL, temperature=0)
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


def calculate_keyword_score(query_keywords: List[str], content: str) -> float:
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


def calculate_bm25_score(query_keywords: List[str], content: str, avg_doc_length: float = 100.0, k1: float = 1.5, b: float = 0.75) -> float:
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
