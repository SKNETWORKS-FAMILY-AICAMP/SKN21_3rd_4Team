"""
Search Router - 검색 전략 결정 모듈

이 모듈의 역할:
- 사용자의 질문을 받아서 분석합니다
- 어디서 검색할지 결정합니다 (lecture DB, python_doc DB, 또는 둘 다)
- 몇 개의 문서를 검색할지 결정합니다
- 어떤 방법으로 검색할지 결정합니다 (similarity, mmr)

핵심 개념: 완전 LLM 기반
- LLM(GPT-4o-mini)이 모든 판단을 수행합니다
- Structured Output으로 안정적인 데이터 반환
- 질문 유효성, 타입, 난이도, 검색 소스를 자동 분석
"""

import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (OPENAI_API_KEY 등)
load_dotenv()

from typing import Dict, List, Literal
from langchain_openai import ChatOpenAI  # OpenAI LLM 인터페이스
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel, Field    # 데이터 검증 및 구조화

from src.agent.prompts import PROMPTS
from src.utils.config import ConfigLLM

def build_search_config(query: str) -> Dict:
    """
    LLM을 활용하여 질문 분석 및 검색 설정을 한 번에 생성합니다.
    
    Args:
        query: 사용자 질문
        
    Returns:
        {
            'sources': List[str],
            'top_k': int,
            'search_method': str,
            'filters': Dict
        }
    """
    class SearchConfig(BaseModel):
        is_valid: bool = Field(
            description="질문이 머신러닝/Python 학습 자료와 관련된 명확하고 구체적인 질문인지 여부. "
                       "반드시 질문 형식이거나 학습 목적이 명확해야 True. "
                       "False인 경우: 의미없는 단어 혼합, 의미없는 반복 단어, 일상 대화/감정 표현, "
                       "키워드만 나열된 경우(실제 질문 아님), 학습과 무관한 주제, 명확하지 않은 질문. "
                       "중요: 키워드('파이썬', '머신러닝')가 있어도 실제 질문이 아니거나 일상 대화면 반드시 False!"
        )
        
        query_type: Literal['concept', 'code', 'syntax'] = Field(
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
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(PROMPTS["SEARCH_ROUTER_PROMPT"])
    ])
    
    llm = ChatOpenAI(
        model=ConfigLLM.OPENAI_MODEL,
        temperature=0
    )
    
    structured_llm = llm.with_structured_output(SearchConfig)
    chain = prompt | structured_llm
    result = chain.invoke({"query": query})

    return {
        'sources': result.search_sources,
        'top_k': result.top_k,
        'search_method': result.search_method,
        'filters': {},
        'is_valid': result.is_valid,
        
        '_analysis': {
            'query_type': result.query_type,
            'topic_keywords': result.topic_keywords,
            'complexity': result.complexity
        }
    }
