# RST 검색 품질 테스트 스크립트
"""
introduction.rst로 ingestion한 데이터의 검색 품질 확인
search_agent와 유사한 구조로 간단하게 테스트
"""
import sys
import os
import time
from pathlib import Path

sys.path.append(os.getcwd())

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient


def test_rst_search():
    load_dotenv(override=True)
    
    # Qdrant 직접 연결
    client = QdrantClient(host="localhost", port=6333)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    collection_name = "learning_ai"
    
    # 테스트 질문들 (한국어 vs 영어 비교)
    # 문서에 있는 표현을 영어 질문으로 사용하여 '상한선' 확인
    # 테스트 질문들 (한글 vs 영어 비교) 확장
    # introduction.rst의 다양한 주제(Numbers, Strings, Lists, First Steps) 커버
    test_pairs = [
        # 1. Numbers
        (
            "파이썬에서 숫자 연산하는 방법", 
            "Using Python as a Calculator numbers operators +, -, *, /"
        ),
        (
            "파이썬 나눗셈 종류", 
            "Division floor division remainder operator"
        ),
        (
            "거듭제곱(승수) 계산하는 방법",
            "calculate powers ** operator squared"
        ),
        (
            "대화형 모드에서 마지막 계산 결과 변수",
            "interactive mode last printed expression variable _"
        ),
        
        # 2. Strings
        (
            "특수문자 이스케이프 무시하는 문자열",
            "raw strings r before the first quote special characters"
        ),
        (
            "문자열 여러 번 반복하기",
            "Strings repeated with * operator"
        ),
        (
            "문자열 슬라이싱 하는 법",
            "string slicing indices substring s[0:2]"
        ),
        
        # 3. Lists
        (
            "리스트에 요소 추가하는 방법", 
            "add new items at the end of the list append method"
        ),
        (
            "리스트의 내용을 변경하는 방법",
            "lists are a mutable type possible to change their content"
        ),
        (
            "리스트 컴프리헨션이란",
            "list comprehension concise way to create lists"
        ),
        
        # 4. First Steps / Control Flow
        (
            "변수 여러 개를 한 번에 할당하기",
            "multiple assignment variables simultaneously get new values"
        ),
        (
            "print 함수에서 줄바꿈 안 하는 방법",
            "print function end keyword argument avoid the newline"
        ),
        (
            "for문에서 range 사용법",
            "for loop range function iterate over a sequence of numbers"
        ),
        (
            "if elif else 조건문 사용법",
            "if elif else statement conditional execution"
        ),
        
        # 5. Functions
        (
            "함수 정의하는 방법",
            "def keyword define function parameters arguments"
        ),
        (
            "람다 함수 사용법",
            "lambda expression anonymous function small functions"
        ),
        (
            "함수에서 기본값 인자 설정",
            "default argument values function definition"
        ),
        
        # 6. Data Structures
        (
            "딕셔너리 사용법",
            "dictionary dict key value pairs mapping type"
        ),
        (
            "튜플과 리스트의 차이",
            "tuple immutable list mutable sequence types"
        ),
        (
            "set 집합 자료형 사용법",
            "set unordered collection duplicate elimination"
        ),
        
        # 7. Modules / Packages
        (
            "모듈 임포트 하는 법",
            "import module from import statement"
        ),
        (
            "패키지란 무엇인가",
            "package __init__.py submodules directory"
        ),
        
        # 8. File I/O
        (
            "파일 읽고 쓰는 방법",
            "open file read write with statement"
        ),
        (
            "with문으로 파일 열기",
            "with statement context manager file handling"
        ),
        
        # 9. Exceptions
        (
            "예외 처리하는 방법",
            "try except exception handling error"
        ),
        (
            "사용자 정의 예외 만들기",
            "raise custom exception class"
        ),
        
        # 10. Classes / OOP
        (
            "클래스 정의하는 방법",
            "class definition object oriented programming"
        ),
        (
            "상속이란 무엇인가",
            "inheritance derived class base class subclass"
        ),
        (
            "__init__ 메서드 역할",
            "__init__ constructor initialize instance attributes"
        ),
    ]
    
    print("=" * 80)
    print("🧪 한글 vs 영어 질문 유사도 비교 테스트")
    print("   가설: 문서는 영어인데 질문이 한글이라 점수가 낮은 것이다.")
    print("=" * 80)
    
    for i, (kor_q, eng_q) in enumerate(test_pairs, 1):
        print(f"\n📌 Case {i}")
        print(f"  🇰🇷 한글: {kor_q}")
        print(f"  🇺🇸 영어: {eng_q}")
        print("-" * 40)
        
        for lang, query in [("KOR", kor_q), ("ENG", eng_q)]:
            try:
                # 1. 질문 벡터화
                query_vector = embedding.embed_query(query)
                
                # 2. 검색
                search_result = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=5
                )
                
                # 3. 전체 결과 출력 (Top 3)
                print(f"\n  [{lang}] 검색 결과 Top 3:")
                print("  " + "-" * 60)
                
                best_score = 0
                for idx, hit in enumerate(search_result.points[:3], 1):
                    score = hit.score
                    source = hit.payload.get('metadata', {}).get('source', 'unknown')
                    content = hit.payload.get('page_content', '')[:200].replace('\n', ' ')
                    
                    if idx == 1:
                        best_score = score
                    
                    print(f"  #{idx} [유사도: {score:.4f}] 소스: {source}")
                    print(f"      📄 내용: {content}...")
                    print()
                
                if lang == "ENG":
                    diff = best_score - last_kor_score
                    if last_kor_score > 0:
                        print(f"  📈 ENG vs KOR 상승폭: +{diff:.4f} ({(diff/last_kor_score)*100:.1f}%)")
                    else:
                        print(f"  📈 ENG vs KOR 상승폭: +{diff:.4f}")
                else:
                    last_kor_score = best_score
                    
            except Exception as e:
                print(f"❌ {lang} 검색 실패: {e}")
                
    print("\n" + "=" * 80)
    return


if __name__ == "__main__":
    test_rst_search()
