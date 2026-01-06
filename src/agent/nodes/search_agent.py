"""
Search Agent - 듀얼 쿼리 검색 시스템

무엇을 하는 파일인가?
- 사용자 질문을 Qdrant(Vector DB)에서 검색해, 관련 문서 조각(top_k)을 가져오는 실행/테스트 스크립트입니다.
- Python 공식문서(RST)는 영어 본문이 대부분이라 한글 질문만으로는 유사도 점수가 낮게 나올 수 있어
  "원문(한글) + 번역(영어)"를 같이 검색해 recall을 올리는 전략(dual query)을 사용합니다.

1) 질문 언어 판별: `is_korean()`
2) 검색 설정 결정: `build_search_config(question)`
   - top_k, sources(lecture/python_doc_rst), search_method 등을 결정
3) 소스별 검색: `search_by_source(query, source, executor, top_k)`
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

# 로컬 실행 시 `src.` import가 깨지지 않게 프로젝트 루트를 path에 추가
sys.path.append(os.getcwd())

from src.agent.nodes.search_router import build_search_config
from src.agent.nodes.search_executor import SearchExecutor

# prompts 경로 변경 대응:
# - 기존: src/agent/prompts.py (단일 파일) 를 exec로 로드
# - 현재: src/agent/prompts/ (패키지) 로 이전됨 → PROMPTS 딕셔너리로 접근
from src.agent.prompts import PROMPTS
TRANSLATE_PROMPT = PROMPTS["TRANSLATE_PROMPT"]
from langchain_openai import ChatOpenAI


# ============================================================
# 듀얼 쿼리 검색 함수
# ============================================================

def is_korean(text: str) -> bool:
    """한글 포함 여부 확인"""
    return bool(re.search(r'[가-힣]', text))


def translate_to_english(question: str) -> str:
    """LLM으로 한글 → 영어 검색 쿼리 변환"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = TRANSLATE_PROMPT.format(question=question)
    return llm.invoke(prompt).content.strip()


def search_by_source(query: str, source: str, executor: SearchExecutor, top_k: int) -> list:
    """특정 소스에서만 검색 (Qdrant 필터 사용)"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    query_vector = executor.embeddings.embed_query(query)
    
    search_result = executor.client.query_points(
        collection_name=executor.collection_name,
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


def execute_dual_query_search(question: str, executor: SearchExecutor) -> tuple:
    """
    소스별 듀얼 쿼리 검색
    
    1. LLM이 top_k 결정 (basic=3, intermediate=5, advanced=7)
    2. lecture/python_doc 각각에서 top_k개씩 검색
    3. 합쳐서 유사도 순 정렬 → 최종 top_k 반환
    
    Returns:
        (results, query_info): 결과 리스트와 쿼리 정보
    """
    all_results = []
    query_info = {"original": question, "translated": None, "queries_used": []}
    
    # LLM이 top_k / sources 결정
    config = build_search_config(question)
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
    lecture_results = search_by_source(question, "lecture", executor, top_k) if "lecture" in sources else []

    # 2) python_doc_rst 검색
    python_results = []
    if "python_doc_rst" in sources:
        if is_korean(question):
            # 2-1) 번역(영어 키워드) 검색이 기본
            english_query = translate_to_english(question)
            query_info["translated"] = english_query
            python_results_en = search_by_source(english_query, "python_doc_rst", executor, top_k)
            for r in python_results_en:
                r["query_type"] = "translated"
            all_results.extend(python_results_en)
            query_info["queries_used"].append(f"번역(python_doc_rst): {english_query}")

            # 2-2) fallback: 번역 결과가 약하면 한글 원문으로도 한 번 더 검색
            best_score = python_results_en[0]["score"] if python_results_en else 0
            if (not python_results_en) or (best_score < PYDOC_FALLBACK_SCORE_THRESHOLD):
                python_results = search_by_source(question, "python_doc_rst", executor, top_k)
        else:
            # 영어 질문이면 원문(영어) 그대로
            python_results = search_by_source(question, "python_doc_rst", executor, top_k)
    else:
        python_results = []
    
    for r in lecture_results + python_results:
        r['query_type'] = 'original'
    all_results.extend(lecture_results + python_results)
    query_info["queries_used"].append(f"원본: {question}")
    
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



# ============================================================
# 테스트 실행
# ============================================================

def run_test():
    """듀얼 쿼리 검색 테스트"""
    
    # 테스트 질문 (영어 + 한글)
    test_questions = [
        # 영어 질문
        # "Using Python as a Calculator numbers operators +, -, *, /",
        # "list comprehension concise way to create lists",
        # "try except exception handling error",
        # "open file read write with statement",
        
        # 한글 질문
        "머신러닝이 뭐야?",
        "결정트리가 뭐야?",
        "경사하강법 개념 알려줘"
        "결정트리와 랜덤포레스트의 차이점이 뭐야?",
        "xgboost 모델에 대해 설명해줘",
        "지도학습이 뭐야?",
        "지도학습 비지도 학습이 뭐야?",
        "모델 불러오는 코드 예제 알려줘."
        # "리스트 컴프리헨션이란",
        # "파이썬 예외처리 방법",
        # "딕셔너리 사용법",
        # "파일 읽고 쓰는 방법",
    ]
    
    executor = SearchExecutor()
    
    print("=" * 70)
    print("🔍 듀얼 쿼리 검색 시스템 테스트")
    print("   한글 질문 → 한글 + 영어 동시 검색")
    print("   영어 질문 → 영어만 검색")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"📌 [{i}/{len(test_questions)}] 질문: {question}")
        print("-" * 70)
        
        start = time.time()
        
        try:
            # 듀얼 쿼리 검색 실행
            results, query_info = execute_dual_query_search(question, executor)
            elapsed = time.time() - start
            
            # 결과 출력
            print(f"⏱️  검색 시간: {elapsed:.2f}초")
            print(f"🔤 원본 쿼리: {query_info['original']}")
            if query_info['translated']:
                print(f"🔄 번역 쿼리: {query_info['translated']}")
            
            print(f"\n📊 검색 결과: {len(results)}개")
            print("-" * 50)
            
            # 상위 5개 미리보기
            is_original_korean = is_korean(query_info['original'])
            
            for j, r in enumerate(results[:5], 1):
                source = r['metadata'].get('source', 'unknown')
                score = r['score']
                query_type = r.get('query_type', '?')
                
                # 쿼리 타입에 따른 이모지
                if query_type == 'original':
                    emoji = "🇰🇷" if is_original_korean else "🇺🇸"
                else:  # translated
                    emoji = "🇺🇸"
                
                preview = r['content'][:100].replace('\n', ' ')
                
                print(f"[{j}] {emoji} 유사도: {score:.4f} | 소스: {source}")
                print(f"    {preview}...")
                
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
