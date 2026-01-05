# Qdrant 검색 로직
"""
🧪 Search Agent 통합 테스트 (Router + Executor 연동)
- search_router.py와 동일한 테스트 질문 사용
- 각 질문마다 Router → Executor 흐름 검증
- 성능(시간) 측정 포함
"""
import sys
import os
import time

sys.path.append(os.getcwd())

from src.agent.nodes.search_router import build_search_config
from src.agent.nodes.search_executor import SearchExecutor


# 실행  python test_integration.py


def run_integration_test():
    # search_router.py와 동일한 테스트 질문들
    test_questions = [
        # [Python 공식문서 테스트] - python_doc DB 저장 확인용
        "머신러닝이 뭐야",
        "경사하강법이 뭐야",
        "Randomforest 설명해줘",
        "딥러닝 설명해줘",
        "import module from import statement",
        "open file read write with statement",
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


if __name__ == "__main__":
    run_integration_test()
