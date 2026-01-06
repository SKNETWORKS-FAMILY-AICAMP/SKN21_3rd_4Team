# ========== 번역 프롬프트 ==========
# 목적: "번역"이 아니라 "검색용 영어 키워드 생성"
# - Python 공식문서(영문 RST)에서 잘 걸리게, 문서에 실제로 등장할 법한 용어/구문을 우선
# - 너무 일반적인 표현(methods, usage 등)만 나오면 검색 점수가 흔들리므로 제약을 강하게 둠
TRANSLATE_PROMPT = """너는 개발 문서를 검색하기 위한 '영어 검색 키워드 생성기'다.
아래 한국어 질문을 Python 공식문서(영어)에서 잘 검색되도록 영어 키워드/구문으로 바꿔라.

규칙:
- 문장/설명 금지. "영어 키워드만" 한 줄로 출력.
- 4~12개의 키워드/구문을 공백으로 구분해 출력.
- 가능한 경우 문서에 실제로 자주 등장하는 정확한 용어를 우선:
  예) list comprehension, dictionary display, dict literal, with statement, file object methods, try except, exception handling
- 너무 일반적인 단어만 단독으로 쓰지 말 것: method(s), usage, thing, how to 등
- 필요하면 문법/토큰을 함께 포함: {{}}, [], (), try, except, with open, KeyError

한국어 질문: {question}
영어 키워드:"""