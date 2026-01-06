# 벡터 검색 품질 테스트 스크립트
"""
lecture와 python_doc_rst 모두 테스트 가능
search_agent와 유사한 구조로 간단하게 테스트

사용법:
1. 아래 EMBEDDING_MODEL 변수를 변경하여 테스트
   - "text-embedding-3-small" (1536 차원)
   - "text-embedding-3-large" (3072 차원)
2. python src/test_vector_search.py 실행

주의:
- 컬렉션의 벡터 크기와 임베딩 모델이 일치해야 함!
- lecture와 python_doc_rst가 같은 컬렉션이면 같은 임베딩 모델 사용 필수
"""
import sys
import os
import time
import argparse
from pathlib import Path

sys.path.append(os.getcwd())

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.utils.config import ConfigDB, ConfigAPI
from src.agent.prompts import PROMPTS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# ============================================================
# 테스트 설정 (여기서 쉽게 변경 가능)
# ============================================================
# 임베딩 모델 선택: "text-embedding-3-small" 또는 "text-embedding-3-large"
EMBEDDING_MODEL = "text-embedding-3-large"  # ← 여기 변경!

# 컬렉션 이름 (None이면 ConfigDB.COLLECTION_NAME 사용)
COLLECTION_NAME = None  # ← 필요시 변경


def get_vector_size(model_name: str) -> int:
    """임베딩 모델에 따른 벡터 크기 반환"""
    if "3-large" in model_name:
        return 3072
    elif "3-small" in model_name:
        return 1536
    else:
        return 1536  # 기본값


def is_korean(text: str) -> bool:
    """한글 포함 여부 확인"""
    return bool(re.search(r'[가-힣]', text))


def create_translate_chain():
    """
    번역용 LangChain chain 생성 (search_agent와 동일)
    
    Returns:
        Chain: prompt | llm | parser 형태의 chain
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(PROMPTS["TRANSLATE_PROMPT"])
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    return chain


def translate_to_english(query: str) -> str:
    """
    LLM으로 한글 → 영어 검색 쿼리 변환 (search_agent와 동일)
    
    Args:
        query: 한글 질문
        
    Returns:
        영어 검색 키워드
    """
    chain = create_translate_chain()
    return chain.invoke({"query": query}).strip()


def test_vector_search(
    embedding_model: str = None,
    collection_name: str = None,
    use_translation: bool = True
):
    """
    벡터 검색 품질 테스트 (lecture + python_doc_rst)
    
    Args:
        embedding_model: 사용할 임베딩 모델 (None이면 파일 상단 EMBEDDING_MODEL 사용)
        collection_name: 사용할 컬렉션 이름 (None이면 파일 상단 COLLECTION_NAME 또는 ConfigDB.COLLECTION_NAME 사용)
        use_translation: 한글 질문을 영어로 번역해서 python_doc_rst 검색할지 여부 (기본: True)
    """
    load_dotenv(override=True)
    
    # 기본값 설정 (파일 상단 변수 사용)
    if embedding_model is None:
        embedding_model = EMBEDDING_MODEL
    if collection_name is None:
        collection_name = COLLECTION_NAME if COLLECTION_NAME else ConfigDB.COLLECTION_NAME
    
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
    print(f"   번역 사용: {use_translation} (한글 질문 → 영어 키워드로 python_doc_rst 검색)")
    print("=" * 80)
    
    # 테스트 질문들 (간단한 한 줄 형식)
    test_querys = [
        # ========== 한글 질문 (lecture 테스트용) ==========
        # "유닛/노드/뉴런 개념 알려줘.",
        # "레이어, 층에 대해서 알려줘.",
        # "입력층이 뭐야?",
        # "머신러닝이 뭐야?",
        # "결정트리가 뭐야?",
        # "경사하강법 개념 알려줘",
        # "결정트리와 랜덤포레스트의 차이점이 뭐야?",
        # "xgboost 모델에 대해 설명해줘",
        # "비지도 학습이 뭐야?",
        
        # ========== 한글 질문 (python_doc_rst 테스트용 - 번역 프롬프트 테스트) ==========
        # Numbers & Operators
        "파이썬에서 숫자 연산하는 방법",
        "정수 나눗셈과 나머지 연산자 사용법",
        # "거듭제곱 연산자 사용하는 방법",
        
        # Strings
        "원시 문자열 리터럴이 뭐야?",
        "문자열 슬라이싱 하는 법",
        # "문자열 메서드 format replace split join 사용법",
        
        # Lists
        "리스트에 요소 추가하는 방법 append extend insert",
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
        "딕셔너리 메서드 get keys values items",
        # "튜플과 리스트의 차이점",
        # "set 집합 자료형 사용법",
        
        # Modules / Packages
        "모듈 임포트 하는 방법",
        "패키지 디렉토리 __init__.py",
        # "from import로 특정 이름만 가져오는 방법",
        
        # File I/O
        "파일 객체 메서드 read write close",
        "with문으로 파일 열기",
        # "파일 읽고 쓰는 방법 텍스트 모드 바이너리 모드",
        
        # Exceptions
        "try except 예외 처리하는 방법",
        "사용자 정의 예외 만드는 방법",
        # "finally 절 사용법",
        
        # Classes / OOP
        "클래스 정의하는 방법",
        "상속이란 무엇인가",
        # "__init__ 메서드 역할",
        # "인스턴스 메서드 클래스 메서드 정적 메서드 차이",
        
        # ========== 영어 질문 (python_doc_rst 테스트용 - RST 문서 용어 사용) ==========
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
    print("🧪 벡터 검색 품질 테스트 (lecture + python_doc_rst)")
    print("=" * 80)
    
    for i, query in enumerate(test_querys, 1):
        print(f"\n{'='*80}")
        print(f"📌 [{i}/{len(test_querys)}] 질문: {query}")
        print("-" * 80)
        
        try:
            all_results = []
            
            # 1. lecture 검색 (원문 그대로 - 한글 문서)
            lecture_query = query  # lecture는 항상 원문(한글)으로 검색
            lecture_vector = embedding.embed_query(lecture_query)
            lecture_result = client.query_points(
                collection_name=collection_name,
                query=lecture_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value="lecture")
                        )
                    ]
                ),
                limit=5
            )
            for hit in lecture_result.points:
                all_results.append({
                    "content": hit.payload.get('page_content', ''),
                    "score": hit.score,
                    "source": hit.payload.get('metadata', {}).get('source', 'unknown'),
                    "query_type": "original"
                })
            
            # 2. python_doc_rst 검색 (번역 사용 시 영어로, 아니면 원문)
            if use_translation and is_korean(query):
                # 한글 질문이면 번역해서 검색
                try:
                    translated_query = translate_to_english(query)
                    print(f"🔄 번역 쿼리: {translated_query}")
                    python_query = translated_query
                except Exception as e:
                    print(f"⚠️  번역 실패: {e} (원문으로 검색)")
                    python_query = query
            else:
                # 영어 질문이거나 번역 비활성화면 원문 그대로
                python_query = query
            
            python_vector = embedding.embed_query(python_query)
            python_result = client.query_points(
                collection_name=collection_name,
                query=python_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value="python_doc_rst")
                        )
                    ]
                ),
                limit=5
            )
            for hit in python_result.points:
                all_results.append({
                    "content": hit.payload.get('page_content', ''),
                    "score": hit.score,
                    "source": hit.payload.get('metadata', {}).get('source', 'unknown'),
                    "query_type": "translated" if (use_translation and is_korean(query) and python_query != query) else "original"
                })
            
            # 3. 결과 정렬 (유사도 순)
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # 4. 결과 출력 (Top 3)
            print(f"\n📊 검색 결과: {len(all_results)}개 (lecture: {len(lecture_result.points)}개, python_doc_rst: {len(python_result.points)}개)")
            print("-" * 50)
            
            for idx, result in enumerate(all_results[:3], 1):
                score = result['score']
                source = result['source']
                query_type = result.get('query_type', 'original')
                content = result['content'][:200].replace('\n', ' ')
                
                emoji = "🇰🇷" if query_type == "original" and is_korean(query) else "🇺🇸"
                print(f"[{idx}] {emoji} 유사도: {score:.4f} | 소스: {source}")
                print(f"    {content}...")
                print()
                    
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            import traceback
            traceback.print_exc()
                
    print("\n" + "=" * 80)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="벡터 검색 품질 테스트 (lecture + python_doc_rst)",
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
        help=f"사용할 임베딩 모델 (None이면 파일 상단 EMBEDDING_MODEL={EMBEDDING_MODEL} 사용)"
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
    
    args = parser.parse_args()
    
    test_vector_search(
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        use_translation=not args.no_translation
    )
