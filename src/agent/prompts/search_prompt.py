# ========== 검색 라우터 프롬프트 ==========
SEARCH_ROUTER_PROMPT = """다음 질문을 분석하고, 최적의 검색 설정을 결정해주세요:

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
