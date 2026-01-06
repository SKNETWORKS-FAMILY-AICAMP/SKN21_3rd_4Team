# Qdrant 검색 로직
"""
  Search Agent 통합 테스트 (Router + Executor 연동)
- search_router.py와 동일한 테스트 질문 사용
- 각 질문마다 Router → Executor 흐름 검증
- 성능(시간) 측정 포함
- 듀얼 쿼리 검색: 한글+영어 동시 검색으로 소스별 균형 확보
"""
import sys
import os
import time
import re

sys.path.append(os.getcwd())

from src.agent.nodes.search_router import build_search_config
from src.agent.nodes.search_executor import SearchExecutor
from src.agent.prompts import TRANSLATE_PROMPT
from langchain_openai import ChatOpenAI


# ========== 듀얼 쿼리 검색 함수들 ==========

def is_korean(text: str) -> bool:
    """한글 포함 여부 확인"""
    return bool(re.search(r'[가-힣]', text))


def translate_to_english(question: str) -> str:
    """LLM으로 한글 → 영어 검색 쿼리 변환 (prompts.py의 TRANSLATE_PROMPT 사용)"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = TRANSLATE_PROMPT.format(question=question)
    return llm.invoke(prompt).content.strip()


def execute_dual_query_search(
    question: str, 
    executor: SearchExecutor, 
    top_k: int = 5
) -> tuple:
    """
    듀얼 쿼리 검색: 한글 질문 → 한글 + 영어 동시 검색
    
    Returns:
        (all_results, query_info): 결과 리스트와 쿼리 정보
    """
    all_results = []
    query_info = {"original": question, "translated": None}
    
    # 1. 원본 쿼리로 검색 (Router 설정 사용)
    config = build_search_config(question)
    original_results = executor.execute_search(question, config)
    for r in original_results:
        r['query_type'] = 'original'
    all_results.extend(original_results)
    
    # 2. 한글이면 영어 번역 후 추가 검색
    if is_korean(question):
        english_query = translate_to_english(question)
        query_info["translated"] = english_query
        
        config_en = build_search_config(english_query)
        english_results = executor.execute_search(english_query, config_en)
        for r in english_results:
            r['query_type'] = 'translated'
        all_results.extend(english_results)
    
    # 3. 중복 제거 (내용 기준)
    seen = set()
    unique_results = []
    for r in all_results:
        content_key = r['content'].strip()[:100]
        if content_key not in seen:
            seen.add(content_key)
            unique_results.append(r)
    
    # 4. 유사도 순 정렬 후 상위 반환
    unique_results.sort(key=lambda x: x['score'], reverse=True)
    
    return unique_results[:top_k * 2], query_info, config

def run_integration_test():
    # search_router.py와 동일한 테스트 질문들
    test_questions = [
        # [Python 공식문서 테스트] - python_doc DB 저장 확인용
        # "머신러닝이 뭐야",
        # "경사하강법이 뭐야",
        # "Randomforest 설명해줘",
        # "딥러닝 설명해줘",
        # "import module from import statement",
        # "open file read write with statement",
        "Using Python as a Calculator numbers operators +, -, *, /",
        "Division floor division remainder operator",
        "open file read write with statement",
        "try except exception handling error",
        "class definition object oriented programming",
        "multiple assignment variables simultaneously get new values",
        "raw strings r before the first quote special characters",
        "파이썬 계산기 숫자 연산자 +, -, *, /",
        "나눗셈 몫 나머지 연산자",
        "파일 열기 읽기 쓰기 with 문",
        "try except 예외 처리 에러",
        "클래스 정의 객체 지향 프로그래밍",
        "다중 할당 변수 동시에 새 값 받기",
        "raw 문자열 r 따옴표 앞 특수 문자",
        
        # "Randomforast 설명해줘",
        # "파이썬 클래스 상속 방법"
        # "파이썬 리스트 메서드 종류 알려줘",     # list append, pop, sort 등
        # "파이썬 딕셔너리 사용법",               # dict 기본
        # "파이썬 for문 range 사용법",            # 반복문 기초
        # "파이썬 클래스 상속 방법",              # OOP 기초
        # "파이썬 예외처리 try except",           # 에러 핸들링
    ]
    
    # Executor 인스턴스 (한 번만 생성)
    executor = SearchExecutor()
    
    print("=" * 70)
    print("🚀 Search Agent 통합 성능 테스트")
    print("   Router(Role A) → Executor(Role B) 연동 검증")
    print("=" * 70)
    
    total_start = time.time()
    results_summary = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"📌 [{i}/{len(test_questions)}] 질문: {question}")
        print("-" * 70)
        
        # ========== 1. Role A (Router) 실행 ==========
        router_start = time.time()
        try:
            config = build_search_config(question)
            router_time = time.time() - router_start
            
            print(f"\n1️⃣ [Role A] Router 결과 (⏱️ {router_time:.2f}초)")
            print(f"   - 검색 대상: {config['sources']}")
            print(f"   - 검색 개수: {config['top_k']}개")
            print(f"   - 검색 방법: {config['search_method']}")
            print(f"   - 분석 정보: {config.get('_analysis', {})}")
            
        except Exception as e:
            print(f"   => ❌ Router 실패: {e}")
            results_summary.append({"question": question, "status": "Router 실패"})
            continue
        
        # ========== 2. Role B (Executor) 실행 ==========
        executor_start = time.time()
        try:
            raw_results = executor.execute_search(question, config)
            deduped = executor.deduplicate_results(raw_results)
            context = executor.build_context(deduped)
            executor_time = time.time() - executor_start
            
            print(f"\n2️⃣ [Role B] Executor 결과 (⏱️ {executor_time:.2f}초)")
            print(f"   - 검색된 문서: {len(raw_results)}개")
            print(f"   - 중복 제거 후: {len(deduped)}개")
            
            # 결과 전체 출력 (줄바꿈 포함)
            print(f"   - 검색 결과(Context):\n{context}")
            
        except Exception as e:
            print(f"   => ❌ Executor 실패: {e}")
            results_summary.append({"question": question, "status": "Executor 실패"})
            continue
        
        # ========== 3. 결과 기록 ==========
        total_time = router_time + executor_time
        results_summary.append({
            "question": question[:30] + "..." if len(question) > 30 else question,
            "sources": config['sources'],
            "docs": len(deduped),
            "router_time": router_time,
            "executor_time": executor_time,
            "total_time": total_time,
            "status": "✅ 성공"
        })
    
    # ========== 최종 요약 ==========
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)
    print(f"{'질문':<35} {'소스':<20} {'문서수':<8} {'시간(초)':<10} {'상태'}")
    print("-" * 70)
    
    for r in results_summary:
        if "total_time" in r:
            print(f"{r['question']:<35} {str(r['sources']):<20} {r['docs']:<8} {r['total_time']:.2f}s      {r['status']}")
        else:
            print(f"{r['question']:<35} {'-':<20} {'-':<8} {'-':<10} {r['status']}")
    
    print("-" * 70)
    print(f"⏱️ 전체 소요 시간: {total_elapsed:.2f}초")
    print(f"✅ 성공: {sum(1 for r in results_summary if r['status'] == '✅ 성공')}/{len(test_questions)}")
    print("=" * 70)


def run_dual_query_test():
    """
    🧪 듀얼 쿼리 검색 테스트
    한글 질문 → 한글 + 영어 동시 검색으로 양쪽 소스에서 균형있게 결과 확보
    """
    # 한글 테스트 질문들
    test_questions = [
        "리스트 컴프리헨션이란",
        "파이썬 예외처리 방법",
        "딕셔너리 사용법",
        "클래스 상속이란",
        "파일 읽고 쓰는 방법",
    ]
    
    executor = SearchExecutor()
    
    print("=" * 70)
    print("🧪 듀얼 쿼리 검색 테스트")
    print("   한글 질문 → 한글 + 영어 동시 검색")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"📌 [{i}/{len(test_questions)}] 질문: {question}")
        print("-" * 70)
        
        start_time = time.time()
        
        try:
            # 듀얼 쿼리 검색 실행
            results, query_info, config = execute_dual_query_search(
                question, executor, top_k=5
            )
            elapsed = time.time() - start_time
            
            # 결과 출력
            print(f"\n⏱️ 검색 시간: {elapsed:.2f}초")
            print(f"🔤 원본 쿼리: {query_info['original']}")
            if query_info['translated']:
                print(f"🔄 번역 쿼리: {query_info['translated']}")
            
            print(f"\n📊 검색 결과: {len(results)}개")
            print("-" * 50)
            
            for j, r in enumerate(results, 1):
                source = r['metadata'].get('source', 'unknown')
                score = r['score']
                query_type = r.get('query_type', 'unknown')
                content_preview = r['content'][:150].replace('\n', ' ')
                
                # 쿼리 타입에 따른 이모지
                emoji = "🇰🇷" if query_type == 'original' else "🇺🇸"
                
                print(f"\n[{j}] {emoji} 유사도: {score:.4f} | 소스: {source}")
                print(f"    📄 {content_preview}...")
                
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ 듀얼 쿼리 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search Agent 테스트")
    parser.add_argument(
        "--mode", 
        choices=["integration", "dual"], 
        default="dual",
        help="테스트 모드: integration(기존), dual(듀얼쿼리)"
    )
    args = parser.parse_args()
    
    if args.mode == "dual":
        run_dual_query_test()
    else:
        run_integration_test()

