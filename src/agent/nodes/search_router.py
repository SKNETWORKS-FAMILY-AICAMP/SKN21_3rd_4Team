"""
작성자 : 신지용
Search Agent Role A - Search Router/Strategy

이 모듈의 역할:
- 사용자의 질문을 받아서 분석
- 어디서 검색할지 결정 (lecture DB, python_doc DB, 또는 둘 다)
- 몇 개의 문서를 검색할지 결정
- 어떤 방법으로 검색할지 결정 (similarity, mmr)
- 최종 검색 설정을 주원님 에게 전달

핵심 개념: 완전 LLM 기반
- LLM(GPT-4o-mini)이 모든 판단을 수행
- Structured Output으로 안정적인 데이터 반환
"""

import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (OPENAI_API_KEY 등)
load_dotenv()

# torch 로딩 문제 해결 (Python 3.13 호환성)
# Warning 메시지를 숨기고 병렬 처리 관련 이슈 방지
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, List, Literal
from langchain_openai import ChatOpenAI  # OpenAI LLM 인터페이스
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel, Field    # 데이터 검증 및 구조화
from src.agent.prompts import PROMPTS


def build_search_config(query: str) -> Dict:
    """
    LLM을 활용하여 질문 분석 및 검색 설정을 한 번에 생성합니다.
    
    Args:
        query: 사용자 질문
        
    Returns:
        {
            'sources': List[str],           # 검색 대상: ['lecture'], ['python_doc'], 또는 둘 다
            'top_k': int,                   # 검색할 문서 개수
            'search_method': str,           # 'similarity' 또는 'mmr'
            'filters': Dict                 # 메타데이터 필터 (향후 확장)
        }
    """
    # ============================================================
    # 1단계: Pydantic 모델 정의 (LLM의 출력 형식 강제)
    # ============================================================
    # Pydantic은 데이터 검증 라이브러리입니다.
    # 여기서는 LLM이 반환할 JSON의 구조를 정의합니다.
    # 이렇게 하면 LLM이 항상 올바른 형식으로 응답하게 됩니다.
    
    class SearchConfig(BaseModel):
        # 질문 유형 (3가지 중 하나만 가능)
        query_type: Literal['concept', 'code', 'syntax'] = Field(
            description="질문 타입: concept(개념 설명), code(코드 작성/디버깅), syntax(문법)"
        )
        
        # 질문에서 추출한 주요 키워드 리스트
        # 예: "RAG와 pandas 활용법" → ['rag', 'pandas']
        topic_keywords: List[str] = Field(
            description="질문에서 추출된 주요 기술 키워드 (예: rag, python, pandas, iris)"
        )
        
        # 질문의 난이도 (3가지 중 하나)
        # 이에 따라 검색 개수(top_k)가 자동 결정됩니다
        complexity: Literal['basic', 'intermediate', 'advanced'] = Field(
            description="질문의 난이도: basic(기초), intermediate(중급), advanced(고급)"
        )
        
        # 검색할 데이터 소스 (핵심!)
        # ['lecture']: ML 강의만
        # ['python_doc']: Python 문서만
        # ['lecture', 'python_doc']: 둘 다 (복합 질문)
        search_sources: List[Literal['lecture', 'python_doc']] = Field(
            description="검색할 데이터 소스 목록. lecture(강의 자료), python_doc(Python 공식 문서)"
        )
        
        # 검색할 문서 개수 (1~10 범위)
        # ge=1: greater than or equal (1 이상)
        # le=10: less than or equal (10 이하)
        top_k: int = Field(
            description="검색할 문서 개수. basic: 3개, intermediate: 5개, advanced: 7개",
            ge=1,
            le=10
        )
        
        # 검색 방법 (2가지 중 하나)
        # similarity: 단순 유사도 (빠름)
        # mmr: Maximum Marginal Relevance (다양성 고려, 느림)
        search_method: Literal['similarity', 'mmr'] = Field(
            description="검색 방법. similarity(단순 유사도), mmr(다양성 고려, 고급 질문에 적합)"
        )
    
    # ============================================================
    # 2단계: LangChain Chain 생성 (체인화)
    # ============================================================
    # ChatPromptTemplate: 프롬프트를 템플릿으로 관리
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(PROMPTS["SEARCH_ROUTER_PROMPT"])
    ])
    
    # ChatOpenAI: OpenAI의 GPT 모델을 사용하기 위한 인터페이스
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 빠르고 저렴한 모델 (gpt-4보다 10배 이상 저렴)
        temperature=0         # 일관된 결과를 위해 0으로 설정
                              # temperature=0: 항상 동일한 질문에 동일한 답변
                              # temperature=1: 매번 다른 답변 (창의적이지만 불안정)
    )
    
    # Structured Output: LLM의 출력을 Pydantic 모델 형식으로 강제
    # 일반 LLM 호출: "자유 텍스트" 반환 → 파싱 오류 가능 ❌
    # Structured Output: SearchConfig 형식으로만 반환 → 안정적 ✅
    structured_llm = llm.with_structured_output(SearchConfig)
    
    # Chain 연결: prompt | structured_llm
    chain = prompt | structured_llm
    

    # 3단계: Chain 실행 및 결과 반환
    # chain.invoke(): 프롬프트를 LLM에 전송하고 결과를 받음
    # 반환값: SearchConfig 객체 (Pydantic 모델)
    result = chain.invoke({"query": query})
    

    # 4단계: Role B가 사용할 형식으로 변환
    # Role B (Search Executor)가 필요한 정보만 추출해서 반환
    return {
        # 핵심 정보 (Role B가 실제로 사용)
        'sources': result.search_sources,        # 어디서 검색할지
        'top_k': result.top_k,                   # 몇 개 가져올지
        'search_method': result.search_method,   # 어떤 방법으로
        'filters': {},                           # 메타데이터 필터 (향후 확장)
        
        # 디버깅/분석용 추가 정보 (선택사항)
        # Role B는 무시해도 되고, 로깅/분석 시 유용
        '_analysis': {
            'query_type': result.query_type,    # 질문 유형
            'topic_keywords': result.topic_keywords,  # 추출된 키워드
            'complexity': result.complexity           # 난이도
        }
    }


# ============================================================
# 테스트 코드 (파일을 직접 실행할 때만 동작)
# ============================================================

if __name__ == "__main__":
    # 다양한 유형의 질문으로 테스트
    test_querys = [
        "RAG가 뭐야?",                              # ML 개념 질문 (basic)
        "딥러닝 모델 최적화 방법",                    # ML 고급 질문 (advanced)
        "iris 데이터셋 불러오는 코드",               # Python 코드 질문 (basic)
        "Python list comprehension 문법",          # Python 문법 질문 (syntax)
        "RAG 구현할 때 pandas DataFrame 활용법"     # 복합 질문 (ML + Python)
    ]
    
    # 테스트 결과 헤더
    print("=" * 80)
    print("Search Router - 완전 LLM 기반 테스트")
    print("=" * 80)
    
    # 각 질문에 대해 검색 설정 생성 및 출력
    for query in test_querys:
        print(f"\n📌 질문: {query}")
        print("-" * 80)
        
        
        config = build_search_config(query)
        
       
        print(f"✅ 검색 대상: {config['sources']}")        # 어디서 검색할지
        print(f"📊 검색 개수: {config['top_k']}개")        # 몇 개 가져올지
        print(f"🔍 검색 방법: {config['search_method']}")  # 어떤 방법으로
        
        # 디버깅용 분석 정보 출력
        analysis = config['_analysis']
        print(f"\n💡 분석 정보:")
        print(f"   - 질문 유형: {analysis['query_type']}")       # concept/code/syntax
        print(f"   - 주요 키워드: {', '.join(analysis['topic_keywords'])}")  # 추출된 키워드
        print(f"   - 난이도: {analysis['complexity']}")             # basic/intermediate/advanced
        print("=" * 80)

