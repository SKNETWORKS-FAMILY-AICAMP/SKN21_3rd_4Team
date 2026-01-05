"""
작성자 : 신지용
Search Agent Role A - Search Router/Strategy
사용자 질문을 분석하고 적절한 검색 전략을 설계하는 모듈
"""

import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# torch 로딩 문제 해결 (Python 3.13 호환성)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, List, Literal
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


def build_search_config(question: str) -> Dict:
    """
    LLM을 활용하여 질문 분석 및 검색 설정을 한 번에 생성합니다. (Main Entry Point)
    
    Args:
        question: 사용자 질문
        
    Returns:
        {
            'sources': List[str],           # 검색 대상: ['lecture'], ['python_doc'], 또는 둘 다
            'top_k': int,                   # 검색할 문서 개수
            'search_method': str,           # 'similarity' 또는 'mmr'
            'filters': Dict                 # 메타데이터 필터 (향후 확장)
        }
    """
    # Pydantic 모델로 구조화된 출력 정의
    class SearchConfig(BaseModel):
        question_type: Literal['concept', 'code', 'syntax'] = Field(
            description="질문 타입: concept(개념 설명), code(코드 작성/디버깅), syntax(문법)"
        )
        topic_keywords: List[str] = Field(
            description="질문에서 추출된 주요 기술 키워드 (예: rag, python, pandas, iris)"
        )
        complexity: Literal['basic', 'intermediate', 'advanced'] = Field(
            description="질문의 난이도: basic(기초), intermediate(중급), advanced(고급)"
        )
        search_sources: List[Literal['lecture', 'python_doc']] = Field(
            description="검색할 데이터 소스 목록. lecture(강의 자료), python_doc(Python 공식 문서)"
        )
        top_k: int = Field(
            description="검색할 문서 개수. basic: 3개, intermediate: 5개, advanced: 7개",
            ge=1,
            le=10
        )
        search_method: Literal['similarity', 'mmr'] = Field(
            description="검색 방법. similarity(단순 유사도), mmr(다양성 고려, 고급 질문에 적합)"
        )
    
    # LLM 초기화 (구조화된 출력)
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 빠르고 저렴한 모델
        temperature=0  # 일관된 결과를 위해 0으로 설정
    )
    
    # Structured output으로 변환
    structured_llm = llm.with_structured_output(SearchConfig)
    
    # 프롬프트
    prompt = f"""다음 질문을 분석하고, 최적의 검색 설정을 결정해주세요:

질문: "{question}"

분석 및 결정 기준:

1. **question_type** (질문 유형):
   - concept: "~가 뭐야?", "설명해줘", "차이점은?" 같은 개념 이해 질문
   - code: "코드 작성해줘", "구현 방법", "에러 해결" 같은 코드 관련 질문
   - syntax: "문법", "사용법", "어떻게 써?" 같은 Python 문법 질문

2. **topic_keywords** (주요 키워드):
   - 머신러닝/딥러닝: rag, embedding, vector, 모델, 학습, 분류, 회귀, sklearn, iris, 결정트리 등
   - Python 기초: list, dict, tuple, set, for, while, if, def, class, pandas, numpy 등
   - 실용적인 기술명을 소문자로 추출 (예: "RAG가 뭐야?" → ['rag'])

3. **complexity** (난이도):
   - basic: 기본 개념, 간단한 질문 ("list가 뭐야?", "iris 데이터셋이란?")
   - intermediate: 비교, 구현, 응용 ("RAG 구현 방법", "pandas로 데이터 전처리")
   - advanced: 최적화, 알고리즘, 성능 튜닝 ("모델 최적화", "대규모 데이터 처리")

4. **search_sources** (검색 대상) - 매우 중요!:
   - ['lecture']: ML/딥러닝 관련 질문 (RAG, embedding, 분류, 회귀, 모델 등)
   - ['python_doc']: 순수 Python 문법/라이브러리 질문 (list, dict, for, pandas 기초 등)
   - ['lecture', 'python_doc']: ML + Python 복합 질문 예시:
     * "RAG 구현할 때 Python list comprehension 사용 방법"
     * "pandas로 iris 데이터 전처리하는 방법"
     * "scikit-learn으로 분류 모델 만들 때 dictionary 활용법"
   
   판단 기준:
   - ML 키워드만 있으면 → ['lecture']
   - Python 문법 키워드만 있으면 → ['python_doc']
   - ML + Python 문법 둘 다 있으면 → ['lecture', 'python_doc']

5. **top_k** (검색 개수):
   - basic: 3개 (간단한 질문은 적은 문서로 충분)
   - intermediate: 5개 (중급 질문은 중간 개수)
   - advanced: 7개 (복잡한 질문은 많은 문서 참조)

6. **search_method** (검색 방법):
   - similarity: basic/intermediate 질문 (단순 유사도 검색)
   - mmr: advanced 질문 (Maximum Marginal Relevance - 다양성 고려)

예시:
- "RAG가 뭐야?" 
  → lecture만, basic, 3개, similarity
  
- "Python list comprehension 문법"
  → python_doc만, basic, 3개, similarity
  
- "RAG 구현할 때 pandas DataFrame 활용법"
  → lecture + python_doc, intermediate, 5개, similarity
  
- "대규모 데이터셋에서 embedding 벡터 최적화"
  → lecture만, advanced, 7개, mmr
"""
    
    # LLM 호출
    result = structured_llm.invoke(prompt)
    
    return {
        'sources': result.search_sources,
        'top_k': result.top_k,
        'search_method': result.search_method,
        'filters': {},  # 향후 주차별/주제별 필터 추가 가능
        # 디버깅용 추가 정보
        '_analysis': {
            'question_type': result.question_type,
            'topic_keywords': result.topic_keywords,
            'complexity': result.complexity
        }
    }


# 테스트용 코드
if __name__ == "__main__":
    # 여러 테스트 케이스
    test_questions = [
        "RAG가 뭐야?",
        "딥러닝 모델 최적화 방법",
        "iris 데이터셋 불러오는 코드",
        "Python list comprehension 문법",
        "RAG 구현할 때 pandas DataFrame 활용법"  # 복합 질문 테스트
    ]
    
    print("=" * 80)
    print("Search Router - 완전 LLM 기반 테스트")
    print("=" * 80)
    
    for question in test_questions:
        print(f"\n📌 질문: {question}")
        print("-" * 80)
        
        config = build_search_config(question)
        
        # 검색 설정 출력
        print(f"✅ 검색 대상: {config['sources']}")
        print(f"📊 검색 개수: {config['top_k']}개")
        print(f"🔍 검색 방법: {config['search_method']}")
        
        # 분석 정보 출력
        analysis = config['_analysis']
        print(f"\n💡 분석 정보:")
        print(f"   - 질문 유형: {analysis['question_type']}")
        print(f"   - 주요 키워드: {', '.join(analysis['topic_keywords'])}")
        print(f"   - 난이도: {analysis['complexity']}")
        print("=" * 80)

