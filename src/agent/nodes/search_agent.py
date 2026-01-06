# Search Agent - 듀얼 쿼리 검색 시스템
"""
한글 질문 → 한글 + 영어 동시 검색으로 양쪽 소스에서 균형있게 결과 확보
영어 질문 → 영어만 검색

실행: python src/agent/nodes/search_agent.py
"""
import sys
import os
import time
import re

sys.path.append(os.getcwd())

from src.agent.nodes.search_router import build_search_config
from src.agent.nodes.search_executor import SearchExecutor
from src.agent.prompts import PROMPTS
from langchain_openai import ChatOpenAI


# ============================================================
# 듀얼 쿼리 검색 함수
# ============================================================

def is_korean(text: str) -> bool:
    """한글 포함 여부 확인"""
    return bool(re.search(r'[가-힣]', text))


def translate_to_english(query: str) -> str:
    """LLM으로 한글 → 영어 검색 쿼리 변환"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PROMPTS["TRANSLATE_PROMPT"].format(query=query)
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


def execute_dual_query_search(query: str, executor: SearchExecutor) -> tuple:
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
    
    # LLM이 top_k 결정
    config = build_search_config(query)
    top_k = config.get('top_k', 5)
    
    # 1. 원본 쿼리로 소스별 검색
    lecture_results = search_by_source(query, "lecture", executor, top_k)
    python_results = search_by_source(query, "python_doc_rst", executor, top_k)
    
    for r in lecture_results + python_results:
        r['query_type'] = 'original'
    all_results.extend(lecture_results + python_results)
    query_info["queries_used"].append(f"원본: {query}")
    
    # 2. 한글이면 영어 번역 후 소스별 검색
    if is_korean(query):
        english_query = translate_to_english(query)
        query_info["translated"] = english_query
        
        lecture_results_en = search_by_source(english_query, "lecture", executor, top_k)
        python_results_en = search_by_source(english_query, "python_doc_rst", executor, top_k)
        
        for r in lecture_results_en + python_results_en:
            r['query_type'] = 'translated'
        all_results.extend(lecture_results_en + python_results_en)
        query_info["queries_used"].append(f"번역: {english_query}")
    
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
    test_querys = [
        # 영어 질문
        "Using Python as a Calculator numbers operators +, -, *, /",
        "list comprehension concise way to create lists",
        "try except exception handling error",
        "open file read write with statement",
        
        # 한글 질문
        "리스트 컴프리헨션이란",
        "파이썬 예외처리 방법",
        "딕셔너리 사용법",
        "파일 읽고 쓰는 방법",
    ]
    
    executor = SearchExecutor()
    
    print("=" * 70)
    print("🔍 듀얼 쿼리 검색 시스템 테스트")
    print("   한글 질문 → 한글 + 영어 동시 검색")
    print("   영어 질문 → 영어만 검색")
    print("=" * 70)
    
    for i, query in enumerate(test_querys, 1):
        print(f"\n{'='*70}")
        print(f"📌 [{i}/{len(test_querys)}] 질문: {query}")
        print("-" * 70)
        
        start = time.time()
        
        try:
            # 듀얼 쿼리 검색 실행
            results, query_info = execute_dual_query_search(query, executor)
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
