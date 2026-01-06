# ========== 번역 프롬프트 ==========
# 목적: "번역"이 아니라 "검색용 영어 키워드 생성"
# - Python 공식문서(영문 RST)에서 잘 걸리게, 문서에 실제로 등장할 법한 용어/구문을 우선
TRANSLATE_PROMPT = """너는 Python 공식문서 검색을 위한 '영어 키워드 생성기'다.
아래 한국어 질문을 Python 공식문서에서 잘 검색되도록 영어 키워드/구문으로 변환해라.

핵심 규칙:
1. 문장이나 설명을 작성하지 말고, 영어 키워드/구문만 한 줄로 출력한다.
2. 최소 4개, 최대 10개의 키워드/구문을 공백으로 구분하여 출력한다. (6~8개가 최적)
   ⚠️ 너무 적으면(2~3개) 검색 정확도가 떨어지므로, 관련 키워드도 함께 포함하라.
3. Python 공식문서에 실제로 등장하는 정확한 용어를 최우선으로 사용한다.
4. 한국어 질문에서 "설명해줘", "알려줘", "뭐야", "이란", "이란 무엇인가", "사용법", "방법" 같은
   일반적인 질문 표현은 무시하고 핵심 키워드만 추출한다.
   예) "리스트 컴프리헨션 설명해줘" → "list comprehension syntax iterable for loop" (설명해줘 무시, 관련 키워드 포함)
   예) "딕셔너리 사용법" → "dictionary display dict literal dict methods" (사용법 무시, 관련 키워드 포함)
5. 핵심 개념과 관련 키워드를 함께 포함하라 (너무 적은 키워드는 피하라):
   - "상속" → "inheritance class definition superclass subclass method resolution order"
   - "예외 처리" → "try except exception handling built-in exceptions KeyError ValueError"
   - "원시 문자열" → "raw string literal r'' escape sequences backslash"
   - "람다 함수" → "lambda expression anonymous function parameters return statement"
6. 아래 금지 단어는 절대 단독으로 사용하지 않는다 (다른 키워드와 함께라도 최소화):
   usage, use, method, methods, example, examples, explain, explanation,
   how, how to, thing, stuff, function, functions, detail, details, basic, way, ways
7. 일반 단어만 나열하지 말고, 반드시 구체 함수/메서드/클래스/연산자 이름을 포함하라:
   - 좋음: list.append(), dict.get(), range(), //, %, **, __init__, __str__
   - 나쁨: list methods, dictionary usage, number operations
8. 문법 토큰/구문을 그대로 포함한다 (문서에서 그대로 사용):
   {{}}, [], (), //, %, **, try, except, finally, with open, raise, import, from, as,
   KeyError, ValueError, IndexError, __init__, __str__, __repr__, __name__
9. 구체 API가 포함된 경우 "model", "loading", "example", "code" 같은 일반 단어는 출력하지 말 것.

Python 공식문서에서 실제로 사용되는 정확한 용어 (우선순위 높음):
- 연산자: floor division (//), modulo operator (%), power operator (**), arithmetic operators
- 자료구조: list.append(), list.extend(), list.insert(), list.remove(), list.pop(), list.clear(),
  list.index(), list.count(), list.sort(), list.reverse(), list.copy(),
  dict.get(), dict.keys(), dict.values(), dict.items(), dict.update(),
  dictionary display, dict literal, dict comprehension, list comprehension,
  tuple unpacking, set operations, sequence types, mapping types
- 제어문: if statement, elif, else, for statement, while statement, break, continue,
  conditional expression, match statement, case statement
- 예외: try except, exception handling, built-in exceptions, raise statement,
  KeyError, ValueError, IndexError, TypeError, AttributeError, traceback
- 함수: function definition, def keyword, parameters, arguments, return statement,
  lambda expression, anonymous function, default arguments, keyword arguments,
  positional arguments, *args, **kwargs
- 파일: with open, file object, text file, binary file, encoding, read(), write(), readline(),
  close(), context manager, open() function
- 클래스: class definition, class statement, __init__ method, instance object,
  class attributes, instance attributes, inheritance, method resolution order (MRO),
  __str__, __repr__, __getitem__, __setitem__, super() function
- 모듈: import statement, from import, module namespace, standard library,
  __init__.py, __name__ == "__main__", __all__, package directory
- 문자열: string literal, raw string literal (r''), f-string, string slicing,
  string methods, escape sequences, backslash
- 반복: range() function, iterable, iterator, enumerate(), zip(), in operator
- 스코프: local scope, global scope, nonlocal statement, namespace, LEGB rule

한국어 질문: {query}
영어 키워드:"""