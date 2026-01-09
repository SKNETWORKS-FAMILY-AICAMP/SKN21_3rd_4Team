"""
벡터 검색 품질 테스트 스크립트

이 모듈의 역할:
- lecture와 python_doc 검색 품질을 테스트하고 결과를 분석합니다
- search_agent와 유사한 구조로 간단하게 테스트할 수 있습니다
- 하이브리드 검색(벡터 + 키워드 매칭 + BM25)을 사용합니다
- 테스트 결과를 JSON, CSV, TXT 형식으로 저장하여 분석 가능합니다

핵심 개념: 검색 품질 평가
- 다양한 질문으로 검색 정확도 테스트
- 번역 기능 활성화/비활성화 테스트
- 하이브리드 검색 점수 분석 (벡터, 키워드, BM25)
- 프롬프트 버전 추적으로 변경 이력 관리
"""
import sys
import os
import time
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.append(os.getcwd())

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.utils.config import ConfigDB, ConfigAPI
from src.utils.search_utils import (
    is_korean,
    translate_to_english,
    calculate_keyword_score,
    calculate_bm25_score
)
from src.agent.prompts import PROMPTS
from src.agent.nodes.search_router import build_search_config
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# ============================================================
# 테스트 설정 (여기서 쉽게 변경 가능)
# ============================================================
# 임베딩 모델과 컬렉션 이름은 ConfigDB에서 가져오거나
# 명령줄 인자로 지정 가능합니다.
# (파일 하단의 argparse 참조)


def get_vector_size(model_name: str) -> int:
    """임베딩 모델에 따른 벡터 크기 반환"""
    if "3-large" in model_name:
        return 3072
    elif "3-small" in model_name:
        return 1536
    else:
        return 1536  # 기본값


def get_file_hash(file_path: str) -> str:
    """파일의 SHA256 해시 반환 (프롬프트 버전 추적용)"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]  # 16자리만
    except Exception:
        return "unknown"


def get_prompt_version() -> Dict[str, Any]:
    """프롬프트 파일 버전 정보 수집"""
    script_dir = Path(__file__).parent
    prompt_file = script_dir / "agent" / "prompts" / "translate_prompt.py"
    
    version_info = {
        "file": str(prompt_file.relative_to(script_dir.parent)),
        "hash": get_file_hash(str(prompt_file)),
        "modified_time": None,
    }
    
    try:
        if prompt_file.exists():
            mtime = prompt_file.stat().st_mtime
            version_info["modified_time"] = datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        pass
    
    return version_info


def get_preprocessing_config() -> Dict[str, Any]:
    """전처리 설정 정보 수집 (ingestion_rst.py에서)"""
    script_dir = Path(__file__).parent
    ingestion_file = script_dir / "ingestion_rst.py"
    
    config = {
        "file": str(ingestion_file.relative_to(script_dir.parent)),
        "hash": get_file_hash(str(ingestion_file)),
    }
    
    # ingestion_rst.py에서 설정값 읽기 (간단한 파싱)
    try:
        if ingestion_file.exists():
            content = ingestion_file.read_text(encoding='utf-8')
            # EMBEDDING_MODEL 추출
            match = re.search(r'EMBEDDING_MODEL\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                config["embedding_model"] = match.group(1)
            # chunk_size, chunk_overlap 추출
            match = re.search(r'chunk_size:\s*int\s*=\s*(\d+)', content)
            if match:
                config["chunk_size"] = int(match.group(1))
            match = re.search(r'chunk_overlap:\s*int\s*=\s*(\d+)', content)
            if match:
                config["chunk_overlap"] = int(match.group(1))
    except Exception as e:
        config["error"] = str(e)
    
    return config



def hybrid_search(
    client: QdrantClient,
    embedding,
    query: str,
    collection_name: str,
    source_filter: str,
    top_k: int = 5,
    candidate_k: Optional[int] = None,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.2,
    bm25_weight: float = 0.2,
    use_bm25: bool = True,
    keywords: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    하이브리드 검색: 벡터 검색 + 키워드 매칭 + BM25 (Sparse 검색)
    
    Args:
        client: Qdrant 클라이언트
        embedding: 임베딩 모델
        query: 검색 쿼리
        collection_name: 컬렉션 이름
        source_filter: 소스 필터 ("lecture" 또는 "python_doc")
        top_k: 최종 반환할 결과 수 (LLM이 설정한 값)
        candidate_k: 벡터 검색 후보 수 (None이면 top_k * 4, 최대 20)
        vector_weight: 벡터 점수 가중치 (기본 0.6)
        keyword_weight: 키워드 매칭 점수 가중치 (기본 0.2)
        bm25_weight: BM25 점수 가중치 (기본 0.2)
        use_bm25: BM25 사용 여부 (기본 True)
        
    Returns:
        하이브리드 점수로 정렬된 결과 리스트
    """
    # candidate_k 자동 계산 (top_k 기반)
    if candidate_k is None:
        candidate_k = min(top_k * 4, 20)
    
    # 1. 벡터 검색으로 후보 가져오기
    query_vector = embedding.embed_query(query)
    vector_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source_filter)
                )
            ]
        ),
        limit=candidate_k
    )
    
    # 2. 쿼리에서 키워드 추출
    if keywords:
        query_keywords = [k.strip() for k in keywords if k.strip()]
        # print(f"DEBUG: Using provided keywords: {query_keywords}")
    else:
        # 쉼표, 세미콜론, 콜론 제거 후 공백으로 분리
        query_cleaned = query.replace(',', ' ').replace(';', ' ').replace(':', ' ')
        query_keywords = [kw.strip() for kw in query_cleaned.split() if len(kw.strip()) > 2]
    
    # 3. 벡터 검색 결과 수집 및 점수 계산
    candidates = []
    bm25_scores = []
    
    for hit in vector_result.points:
        content = hit.payload.get('page_content', '')
        vector_score = hit.score
        keyword_score = calculate_keyword_score(query_keywords, content)
        
        # BM25 점수 계산 (Sparse 검색)
        bm25_score = 0.0
        if use_bm25:
            bm25_raw = calculate_bm25_score(query_keywords, content)
            bm25_scores.append(bm25_raw)
            # BM25 점수는 나중에 정규화할 예정
        else:
            bm25_scores.append(0.0)
        
        candidates.append({
            "content": content,
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "bm25_raw": bm25_raw if use_bm25 else 0.0,
            "source": hit.payload.get('metadata', {}).get('source', 'unknown'),
            "metadata": hit.payload.get('metadata', {})
        })
    
    # BM25 점수 정규화 (0~1 범위로)
    if use_bm25 and bm25_scores and max(bm25_scores) > 0:
        max_bm25 = max(bm25_scores)
        for i, candidate in enumerate(candidates):
            candidate['bm25_score'] = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0.0
    else:
        for candidate in candidates:
            candidate['bm25_score'] = 0.0
    
    # 하이브리드 점수 계산 (벡터 + 키워드 + BM25)
    for candidate in candidates:
        if use_bm25:
            hybrid_score = (
                candidate['vector_score'] * vector_weight +
                candidate['keyword_score'] * keyword_weight +
                candidate['bm25_score'] * bm25_weight
            )
        else:
            # BM25 없이 기존 방식
            hybrid_score = (
                candidate['vector_score'] * vector_weight +
                candidate['keyword_score'] * keyword_weight
            )
        candidate['score'] = hybrid_score
        
    # 4. 하이브리드 점수로 정렬
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 5. top_k만 반환 (점수 정보 포함)
    return candidates[:top_k]




def save_test_results(
    test_results: List[Dict[str, Any]],
    embedding_model: str,
    collection_name: str,
    use_translation: bool,
    save_dir: Optional[Path] = None
):
    """테스트 결과를 JSON과 CSV로 저장"""
    if save_dir is None:
        script_dir = Path(__file__).parent
        save_dir = script_dir.parent / "results" / "vector_search"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 메타데이터 수집
    prompt_version = get_prompt_version()
    preprocessing_config = get_preprocessing_config()
    
    # 평균 유사도 계산
    valid_scores = [r.get("top_score", 0) for r in test_results if "error" not in r and r.get("top_score")]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    # 전체 결과 구조
    full_results = {
        "metadata": {
            "timestamp": timestamp,
            "embedding_model": embedding_model,
            "collection_name": collection_name,
            "use_translation": use_translation,
            "use_hybrid": True,  # 항상 하이브리드 검색 사용
            "prompt_version": prompt_version,
            "preprocessing_config": preprocessing_config,
            "total_queries": len(test_results),
            "avg_top_score": avg_score,
        },
        "results": test_results
    }
    
    # JSON 저장
    json_path = save_dir / f"vector_search_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과 저장: {json_path}")
    
    # CSV 저장 (간단 버전)
    import csv
    csv_path = save_dir / f"vector_search_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "is_korean", "translated_query", "top_score", 
                       "lecture_count", "python_doc_count", "translation_error"]
        )
        writer.writeheader()
        for r in test_results:
            if "error" not in r:
                writer.writerow({
                    "query": r.get("query", ""),
                    "is_korean": r.get("is_korean", False),
                    "translated_query": r.get("translated_query", ""),
                    "top_score": r.get("top_score", 0.0),
                    "lecture_count": r.get("lecture_count", 0),
                    "python_doc_count": r.get("python_doc_count", 0),
                    "translation_error": r.get("translation_error", "")
                })
    print(f"💾 CSV 저장: {csv_path}")
    
    # 요약 리포트 저장
    report_path = save_dir / f"vector_search_{timestamp}_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("벡터 검색 품질 테스트 결과 요약\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"실행 시간: {timestamp}\n")
        f.write(f"임베딩 모델: {embedding_model}\n")
        f.write(f"컬렉션: {collection_name}\n")
        f.write(f"번역 사용: {use_translation}\n")
        f.write(f"하이브리드 검색: 활성화 (벡터 + 키워드 매칭 + BM25)\n\n")
        f.write(f"프롬프트 버전:\n")
        f.write(f"  파일: {prompt_version.get('file', 'N/A')}\n")
        f.write(f"  해시: {prompt_version.get('hash', 'N/A')}\n")
        f.write(f"  수정 시간: {prompt_version.get('modified_time', 'N/A')}\n\n")
        f.write(f"전처리 설정:\n")
        for k, v in preprocessing_config.items():
            if k != "file" and k != "hash":
                f.write(f"  {k}: {v}\n")
        f.write(f"\n평균 유사도: {avg_score:.4f}\n")
        f.write(f"총 질문 수: {len(test_results)}\n\n")
        f.write("-" * 80 + "\n")
        f.write("질문별 결과:\n")
        f.write("-" * 80 + "\n")
        for i, r in enumerate(test_results, 1):
            if "error" not in r:
                f.write(f"\n[{i}] {r.get('query', '')}\n")
                f.write(f"    번역: {r.get('translated_query', 'N/A')}\n")
                f.write(f"    최고 유사도: {r.get('top_score', 0.0):.4f}\n")
            else:
                f.write(f"\n[{i}] {r.get('query', '')} - ERROR: {r.get('error', '')}\n")
    print(f"💾 요약 리포트 저장: {report_path}")


def test_vector_search(
    embedding_model: str = None,
    collection_name: str = None,
    use_translation: bool = True,
    save_results: bool = True
):
    """
    벡터 검색 품질 테스트 (lecture + python_doc)
    하이브리드 검색(벡터 + 키워드 매칭 + BM25)을 기본으로 사용합니다.
    
    Args:
        embedding_model: 사용할 임베딩 모델 (None이면 파일 상단 EMBEDDING_MODEL 사용)
        collection_name: 사용할 컬렉션 이름 (None이면 파일 상단 COLLECTION_NAME 또는 ConfigDB.COLLECTION_NAME 사용)
        use_translation: 한글 질문을 영어로 번역해서 python_doc 검색할지 여부 (기본: True)
    """
    load_dotenv(override=True)
    
    # 기본값 설정 (ConfigDB 사용)
    if embedding_model is None:
        embedding_model = ConfigDB.EMBEDDING_MODEL
    if collection_name is None:
        collection_name = ConfigDB.COLLECTION_NAME
    
    # Qdrant 직접 연결
    client = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT)
    
    # 임베딩 모델 설정
    embedding = OpenAIEmbeddings(
        model=embedding_model,
        api_key=ConfigAPI.OPENAI_API_KEY
    )
    
    # 벡터 크기 확인
    vector_size = get_vector_size(embedding_model)
    
    print("=" * 80)
    print(f"🔧 설정 정보")
    print(f"   임베딩 모델: {embedding_model}")
    print(f"   벡터 크기: {vector_size}")
    print(f"   컬렉션: {collection_name}")
    print(f"   번역 사용: {use_translation} (한글 질문 → 영어 키워드로 python_doc 검색)")
    print(f"   하이브리드 검색: 활성화 (벡터 + 키워드 매칭 + BM25)")
    print("=" * 80)
    
    # 테스트 질문들 (간단한 한 줄 형식) - lecture + python_doc 모두 포함
    test_querys = [
        # ========== 한글 질문 (lecture 테스트용) ==========
        "유닛/노드/뉴런 개념 알려줘.",
        "레이어, 층에 대해서 알려줘.",
        "입력층이 뭐야?",
        "머신러닝이 뭐야?",
        "결정트리가 뭐야?",
        "경사하강법 개념 알려줘",
        "결정트리와 랜덤포레스트의 차이점이 뭐야?",
        "xgboost 모델에 대해 설명해줘",
        "비지도 학습이 뭐야?",
        "랜덤포레스트가 뭐야?",
        
        # ========== 한글 질문 (python_doc 테스트용 - 번역 프롬프트 테스트) ==========
        # Numbers & Operators
        "파이썬에서 숫자 연산하는 방법",
        "정수 나눗셈과 나머지 연산자 사용법",
        # "거듭제곱 연산자 사용하는 방법",
        
        # Strings
        # "원시 문자열 리터럴이 뭐야?",
        "문자열 슬라이싱 하는 법",
        # "문자열 메서드 format replace split join 사용법",
        
        # Lists
        # "리스트에 요소 추가하는 방법 append extend insert",
        "리스트 컴프리헨션이란",
        # "리스트 요소 수정하는 방법",
        
        # Control Flow
        "if elif else 조건문 사용법",
        "for문에서 range 함수 사용하는 방법",
        # "while문에서 break continue 사용법",
        # "변수 여러 개를 한 번에 할당하는 방법",
        
        # Functions
        "함수 정의하는 방법 def 키워드",
        "람다 함수 사용법",
        # "함수에서 기본값 인자 설정하는 방법",
        # "키워드 인자와 위치 인자 차이",
        
        # Data Structures
        "딕셔너리 리터럴 사용법",
        # "딕셔너리 메서드 get keys values items",
        # "튜플과 리스트의 차이점",
        # "set 집합 자료형 사용법",
        
        # Modules / Packages
        "모듈 임포트 하는 방법",
        # "패키지 디렉토리 __init__.py",
        # "from import로 특정 이름만 가져오는 방법",
        
        # File I/O
        # "파일 객체 메서드 read write close",
        # "with문으로 파일 열기",
        # "파일 읽고 쓰는 방법 텍스트 모드 바이너리 모드",
        
        # Exceptions
        "try except 예외 처리하는 방법",

        # "사용자 정의 예외 만드는 방법",
        # "finally 절 사용법",
        
        # Classes / OOP
        # "클래스 정의하는 방법",
        # "상속이란 무엇인가",
        # "__init__ 메서드 역할",
        # "인스턴스 메서드 클래스 메서드 정적 메서드 차이",
        
        # ========== 영어 질문 (python_doc 테스트용 - RST 문서 용어 사용) ==========
        # Numbers & Operators
        # "Python numbers operators addition subtraction multiplication division",
        # "integer division floor division remainder modulo operator",
        # "power exponentiation operator **",
        
        # # Strings
        # "raw string literal escape sequences r prefix",
        # "string slicing indexing substring",
        # "string methods format replace split join",
        
        # # Lists
        # "list methods append extend insert remove",
        # "list comprehension concise way create lists",
        # "list slicing indexing modify elements",
        
        # # Control Flow
        # "if elif else conditional statements",
        # "for loop range function iterate",
        # "while loop break continue statements",
        # "multiple assignment tuple unpacking",
        
        # # Functions
        # "function definition def keyword parameters",
        # "lambda function anonymous function expression",
        # "default argument values function parameters",
        # "keyword arguments positional arguments",
        
        # # Data Structures
        # "dictionary display dict literal key value pairs",
        # "dict methods get keys values items",
        # "tuple list difference immutable mutable",
        # "set data type unordered unique elements",
        
        # # Modules / Packages
        # "import statement module import",
        # "package directory __init__.py",
        # "from import statement specific names",
        
        # # File I/O
        # "file object methods read write close",
        # "with statement context manager open file",
        # "file reading writing text mode binary mode",
        
        # # Exceptions
        # "try except exception handling error",
        # "raise exception custom exception",
        # "finally clause cleanup code",
        
        # # Classes / OOP
        # "class definition class keyword",
        # "inheritance base class derived class",
        # "__init__ method constructor initialization",
        # "instance method class method static method",
    ]
    
    print("\n" + "=" * 80)
    print("🧪 벡터 검색 품질 테스트 (lecture + python_doc)")
    print("=" * 80)
    
    # 결과 수집용
    test_results = []
    
    for i, query in enumerate(test_querys, 1):
        print(f"\n{'='*80}")
        print(f"📌 [{i}/{len(test_querys)}] 질문: {query}")
        print("-" * 80)
        
        try:
            all_results = []
            lecture_count = 0
            python_doc_count = 0
            
            # LLM이 top_k와 sources 결정 (search_agent와 동일한 로직)
            try:
                config = build_search_config(query)
                top_k = config.get('top_k', 5)
                sources = config.get("sources", ["lecture", "python_doc"])
                analysis = config.get('_analysis', {})
                topic_keywords = analysis.get('topic_keywords', [])
                
                print(f"🤖 LLM 결정: top_k={top_k}개, sources={sources} (complexity: {analysis.get('complexity', 'unknown')})")
                if topic_keywords:
                    print(f"🔑 추출 키워드: {topic_keywords}")
            except Exception as e:
                print(f"⚠️  LLM 설정 결정 실패: {e} (기본값 사용)")
                top_k = 5
                sources = ["lecture", "python_doc"] if is_korean(query) else ["python_doc"]
                topic_keywords = []
            
            # 1. lecture 검색 (LLM이 결정한 sources에 포함되어 있을 때만)
            lecture_count = 0
            if "lecture" in sources:
                lecture_query = query  # lecture는 원문으로 검색
                lecture_results = hybrid_search(
                    client, embedding, lecture_query, collection_name, "lecture", top_k=top_k, use_bm25=True,
                    keywords=topic_keywords
                )
                lecture_count = len(lecture_results)
                for r in lecture_results:
                    all_results.append({
                        "content": r['content'],
                        "score": r['score'],
                        "source": r['source'],
                        "query_type": "original",
                        "vector_score": r.get('vector_score', 0),
                        "keyword_score": r.get('keyword_score', 0),
                        "bm25_score": r.get('bm25_score', 0)
                    })
            
            # 2. python_doc 검색 (LLM이 결정한 sources에 포함되어 있을 때만)
            translated_query = None
            translation_error = None
            python_doc_count = 0
            
            if "python_doc" in sources:
                if is_korean(query):
                    # 한글 질문이면 번역해서 python_doc에서 검색
                    if use_translation:
                        try:
                            translated_query = translate_to_english(query)
                            print(f"🔄 번역 쿼리: {translated_query}")
                            python_query = translated_query
                        except Exception as e:
                            translation_error = str(e)
                            print(f"⚠️  번역 실패: {e} (원문으로 검색)")
                            python_query = query
                    else:
                        python_query = query
                else:
                    # 영어 질문이면 원문 그대로 python_doc에서 검색
                    python_query = query
                
                python_results = hybrid_search(
                    client, embedding, python_query, collection_name, "python_doc", top_k=top_k, use_bm25=True
                )
                python_doc_count = len(python_results)
                query_type = "translated" if (use_translation and is_korean(query) and translated_query) else "original"
                for r in python_results:
                    all_results.append({
                        "content": r['content'],
                        "score": r['score'],
                        "source": r['source'],
                        "query_type": query_type,
                        "vector_score": r.get('vector_score', 0),
                        "keyword_score": r.get('keyword_score', 0),
                        "bm25_score": r.get('bm25_score', 0)
                    })
            
            # 3. 중복 제거 (search_agent와 동일한 로직)
            seen = set()
            unique_results = []
            for r in all_results:
                content_key = r['content'].strip()[:100]
                if content_key not in seen:
                    seen.add(content_key)
                    unique_results.append(r)
            
            # 4. 유사도 순 정렬 후 top_k만 반환 (search_agent와 동일)
            unique_results.sort(key=lambda x: x['score'], reverse=True)
            final_results = unique_results[:top_k]
            
            # 5. 결과 출력 (Top 3)
            top_score = final_results[0]['score'] if final_results else 0.0
            print(f"\n📊 검색 결과: {len(final_results)}개 (lecture: {lecture_count}개, python_doc: {python_doc_count}개)")
            print("-" * 50)
            
            for idx, result in enumerate(final_results[:3], 1):
                score = result['score']
                source = result['source']
                query_type = result.get('query_type', 'original')
                content = result['content'][:200].replace('\n', ' ')
                
                emoji = "🇰🇷" if query_type == "original" and is_korean(query) else "🇺🇸"
                vector_score = result.get('vector_score', 0)
                keyword_score = result.get('keyword_score', 0)
                bm25_score = result.get('bm25_score', 0)
                print(f"[{idx}] {emoji} 하이브리드: {score:.4f} (벡터: {vector_score:.4f}, 키워드: {keyword_score:.4f}, BM25: {bm25_score:.4f}) | 소스: {source}")
                print(f"    {content}...")
                print()
            
            # 결과 저장용 데이터 수집
            test_results.append({
                "query": query,
                "is_korean": is_korean(query),
                "translated_query": translated_query,
                "translation_error": translation_error,
                "top_score": top_score,
                "top_k": top_k,
                "sources": sources,
                "lecture_count": lecture_count,
                "python_doc_count": python_doc_count,
                "top_3_results": [
                    {
                        "score": r['score'],
                        "source": r['source'],
                        "query_type": r.get('query_type', 'original'),
                        "content_preview": r['content'][:200].replace('\n', ' ')
                    }
                    for r in final_results[:3]
                ]
            })
                    
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            import traceback
            traceback.print_exc()
            test_results.append({
                "query": query,
                "error": str(e)
            })
    
    # 결과 저장
    if test_results and save_results:
        save_test_results(
            test_results=test_results,
            embedding_model=embedding_model,
            collection_name=collection_name,
            use_translation=use_translation
        )
    
    print("\n" + "=" * 80)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="벡터 검색 품질 테스트 (lecture + python_doc)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 파일 상단 EMBEDDING_MODEL 변수 사용 (기본)
  python src/test_vector_search.py
  
  # 명령줄로 모델 지정 (파일 설정 무시)
  python src/test_vector_search.py --embedding-model text-embedding-3-small
  
  # 다른 컬렉션 사용
  python src/test_vector_search.py --collection learning_ai_rst_v2
        """
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        choices=["text-embedding-3-small", "text-embedding-3-large"],
        help=f"사용할 임베딩 모델 (None이면 파일 상단 EMBEDDING_MODEL 또는 ConfigDB.EMBEDDING_MODEL 사용)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"사용할 컬렉션 이름 (None이면 파일 상단 COLLECTION_NAME 또는 ConfigDB.COLLECTION_NAME 사용)"
    )
    parser.add_argument(
        "--no-translation",
        action="store_true",
        help="번역 기능 비활성화 (원문 그대로 검색)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="결과 저장 비활성화"
    )
    args = parser.parse_args()
    
    test_vector_search(
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        use_translation=not args.no_translation,
        save_results=not args.no_save
    )
