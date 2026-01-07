# Legacy Vector Search Tool (For "Before" Comparison)
"""
이 스크립트는 "Before" 상태(단순 벡터 검색)를 테스트하기 위해 생성되었습니다.
최신 기능인 Hybrid Search, 검색어 번역, LLM Router 등을 모두 제외하고
오직 '임베딩 -> 벡터 검색'만 수행합니다.

결과는 results/vector_search/legacy_{category}_{timestamp}.csv 에 저장됩니다.
"""

import sys
import os
import argparse
import csv
import json
import datetime
from typing import List, Dict, Any
from pathlib import Path

sys.path.append(os.getcwd())

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from src.utils.config import ConfigDB, ConfigAPI

DOC_QUERIES = [
    # Numbers & Operators
    "파이썬에서 숫자 연산하는 방법",
    
    # Strings
    "문자열 슬라이싱 하는 법",
    
    # Lists
    "리스트 컴프리헨션이란",
    
    # Control Flow
    "if elif else 조건문 사용법",

    # Functions
    "람다 함수 사용법",
    
    # Data Structures
    "딕셔너리 리터럴 사용법",
    
    # Modules / Packages
    "from import로 특정 이름만 가져오는 방법",
    
    # File I/O
    "파일 객체 메서드 read write close",
    
    # Exceptions
    "try except 예외 처리하는 방법",
    
    # Classes / OOP
    "상속이란 무엇인가",
]

LECTURE_QUERIES = [
    # "유닛/노드/뉴런 개념 알려줘.",
    # "레이어, 층에 대해서 알려줘.",
    # "입력층이 뭐야?",
    # "머신러닝이 뭐야?",
    # "결정트리가 뭐야?",
    # "경사하강법 개념 알려줘",
    # "결정트리와 랜덤포레스트의 차이점이 뭐야?",
    # "xgboost 모델에 대해 설명해줘",
    # "비지도 학습이 뭐야?",
    # "랜덤포레스트가 뭐야?",
]

def simple_vector_search(
    client: QdrantClient,
    embedding,
    query: str,
    collection_name: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    단순 벡터 검색 (Hybrid X, Keyword X, BM25 X)
    """
    query_vector = embedding.embed_query(query)
    vector_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
    
    results = []
    for hit in vector_result.points:
        results.append({
            "content": hit.payload.get('page_content', '') or hit.payload.get('content', ''),
            "score": hit.score,
            "source": hit.payload.get('metadata', {}).get('source', 'unknown'),
            "metadata": hit.payload
        })
    return results

def save_results(results_data: List[Dict], category: str, collection_name: str, all_query_results: List[Dict]):
    """결과를 JSON과 CSV로 저장 (최소한의 형식)"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "vector_search_legacy"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 평균 유사도 계산
    valid_scores = [r.get("top_score", 0) for r in all_query_results if "error" not in r and r.get("top_score")]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    # JSON 저장 (메타데이터 + 결과)
    json_filepath = output_dir / f"legacy_{category}_{timestamp}.json"
    full_results = {
        "metadata": {
            "timestamp": timestamp,
            "embedding_model": "text-embedding-3-small",
            "collection_name": collection_name,
            "use_translation": False,
            "use_hybrid": False,
            "total_queries": len(all_query_results),
            "avg_top_score": avg_score,
        },
        "results": all_query_results
    }
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON 저장: {json_filepath}")
    
    # CSV 저장 (간단한 요약)
    csv_filepath = output_dir / f"legacy_{category}_{timestamp}.csv"
    with open(csv_filepath, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "is_korean", "top_score", "lecture_count", "python_doc_count"]
        )
        writer.writeheader()
        for r in all_query_results:
            if "error" not in r:
                writer.writerow({
                    "query": r.get("query", ""),
                    "is_korean": r.get("is_korean", False),
                    "top_score": r.get("top_score", 0.0),
                    "lecture_count": r.get("lecture_count", 0),
                    "python_doc_count": r.get("python_doc_count", 0)
                })
    print(f"💾 CSV 저장: {csv_filepath}")

def run_legacy_test(collection_name: str, category: str):
    load_dotenv(override=True)
    
    print(f"🔧 Legacy Search Test (Collection: {collection_name}, Category: {category})")
    print("-" * 60)

    try:
        client = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT)
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=ConfigAPI.OPENAI_API_KEY
        )
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # Select queries
    if category == "python_doc":
        test_queries = DOC_QUERIES
    elif category == "lecture":
        test_queries = LECTURE_QUERIES
    else:
        test_queries = DOC_QUERIES + LECTURE_QUERIES

    all_csv_rows = []
    all_query_results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] 질문: {query}")
        try:
            results = simple_vector_search(client, embedding, query, collection_name, top_k=5)
            if not results:
                print("   >> 검색 결과 없음")
                all_query_results.append({
                    "query": query,
                    "error": "검색 결과 없음"
                })
                continue
            
            top_score = results[0]['score'] if results else 0.0
            
            # 소스별 카운트
            lecture_count = sum(1 for r in results if r.get('source') == 'lecture')
            python_doc_count = sum(1 for r in results if r.get('source') == 'python_doc')
                
            for rank, r in enumerate(results[:3], 1):
                content_preview = r['content'][:100].replace('\n', ' ')
                print(f"   {rank}. [{r['score']:.4f}] {content_preview}...")
                
                # CSV Row Data
                all_csv_rows.append({
                    "query": query,
                    "rank": rank,
                    "score": r['score'],
                    "content": r['content'],
                    "source": r['source'],
                    "metadata": str(r['metadata'])
                })
            
            # JSON용 결과 데이터
            all_query_results.append({
                "query": query,
                "is_korean": bool(any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in query)),
                "top_score": top_score,
                "lecture_count": lecture_count,
                "python_doc_count": python_doc_count,
                "top_3_results": [
                    {
                        "score": r['score'],
                        "source": r['source'],
                        "content_preview": r['content'][:200].replace('\n', ' ')
                    }
                    for r in results[:3]
                ]
            })
        except Exception as e:
            print(f"   >> 검색 도중 에러: {e}")
            all_query_results.append({
                "query": query,
                "error": str(e)
            })
            
    # Save results
    if all_csv_rows:
        save_results(all_csv_rows, category, collection_name, all_query_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legacy Vector Search Test")
    parser.add_argument("--collection", type=str, default="learning_ai_legacy", help="Collection name for legacy data")
    parser.add_argument("--category", type=str, default="all", choices=["python_doc", "lecture", "all"], 
                       help="Test category (기본값: all = lecture + python_doc 둘 다)")
    args = parser.parse_args()
    
    # 기본값: lecture + python_doc 둘 다 테스트
    run_legacy_test(args.collection, args.category)
