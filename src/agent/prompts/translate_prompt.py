# ========== 번역 프롬프트 ==========
# 목적: "번역"이 아니라 "검색용 영어 키워드 생성"
# - Python 공식문서(영문 RST)에서 잘 걸리게, 문서에 실제로 등장할 법한 용어/구문을 우선
TRANSLATE_PROMPT = """너는 Python 공식문서 검색을 위한 '영어 키워드 생성기'다.
아래 한국어 질문을 Python 공식문서에서 잘 검색되도록 영어 키워드/구문으로 변환해라.

⚠️ 필수 규칙 (반드시 준수):
1. 문장이나 설명을 작성하지 말고, 영어 키워드/구문만 **공백으로만 구분하여** 한 줄로 출력한다.
   ⚠️ 쉼표(,), 세미콜론(;), 콜론(:)은 절대 사용하지 말 것! 공백으로만 구분!
2. 🔴 반드시 최소 4개 이상의 키워드를 출력해야 한다. 2~3개는 절대 안 된다!
   - 최소 4개, 최대 10개 (6~8개가 최적)
   - 키워드가 부족하면 관련 개념, 메서드, 연산자, 문법 토큰을 추가하라
   - 예) "원시 문자열" → "raw string literal r'' escape sequences backslash" (4개 이상)
   - 예) "상속" → "inheritance class definition superclass subclass method resolution order" (6개)
3. Python 공식문서에 실제로 등장하는 정확한 용어를 최우선으로 사용한다.
4. 한국어 질문에서 "설명해줘", "알려줘", "뭐야", "이란", "이란 무엇인가", "사용법", "방법" 같은
   일반적인 질문 표현은 무시하고 핵심 키워드만 추출한다.
5. 아래 금지 단어는 절대 단독으로 사용하지 않는다 (다른 키워드와 함께라도 최소화):
   usage, use, method, methods, example, examples, explain, explanation,
   how, how to, thing, stuff, function, functions, detail, details, basic, way, ways
6. 일반 단어만 나열하지 말고, 반드시 구체 함수/메서드/클래스/연산자 이름을 포함하라:
   - 좋음: list.append(), dict.get(), range(), //, %, **, __init__, __str__
   - 나쁨: list methods, dictionary usage, number operations
7. 문법 토큰/구문을 그대로 포함한다 (문서에서 그대로 사용):
   {{}}, [], (), //, %, **, try, except, finally, with open, raise, import, from, as,
   KeyError, ValueError, IndexError, __init__, __str__, __repr__, __name__
8. 구체 API가 포함된 경우 "model", "loading", "example", "code" 같은 일반 단어는 출력하지 말 것.

핵심 개념별 필수 키워드 매핑 (반드시 포함):
아래 개념이 질문에 포함되면, 반드시 해당 필수 키워드를 포함해야 한다:

- "상속" / "inheritance" → 반드시 포함: "method resolution order" 또는 "MRO"
- "원시 문자열" / "raw string" → 반드시 포함: "escape sequences"
- "__init__" / "생성자" / "초기화" → 반드시 포함: "__init__"
- "예외" / "exception" → 반드시 포함: "try except" 또는 구체 예외명 (KeyError, ValueError 등)
- "클래스" / "class" → 반드시 포함: "class definition" 또는 "class statement"
- "모듈" / "module" → 반드시 포함: "import statement" 또는 "from import"
- "딕셔너리" / "dictionary" → 반드시 포함: "dict literal" 또는 "dictionary display" 또는 "dict.get()"
- "리스트" / "list" → 반드시 포함: "list.append()" 또는 "list comprehension" 또는 구체 메서드명
- "문자열" / "string" → 반드시 포함: "string literal" 또는 "string slicing" 또는 구체 메서드명
- "함수" / "function" → 반드시 포함: "def keyword" 또는 "function definition"
- "람다" / "lambda" → 반드시 포함: "lambda expression" 또는 "anonymous function"

구체적인 변환 예시 (반드시 참고):
- "원시 문자열 리터럴이 뭐야?" 
  → "raw string literal r'' escape sequences backslash string literal" (6개, escape sequences 필수 포함)
  
- "상속이란 무엇인가"
  → "inheritance class definition superclass subclass method resolution order MRO" (7개, method resolution order 필수 포함)
  
- "사용자 정의 예외 만드는 방법"
  → "raise exception custom exception class definition __init__ exception handling built-in exceptions" (7개, __init__ 필수 포함)
  
- "리스트 컴프리헨션 설명해줘"
  → "list comprehension syntax iterable for loop expression brackets []" (6개, 구체 문법 포함)
  
- "딕셔너리 리터럴 사용법"
  → "dictionary display dict literal key value pairs curly braces {{}} dict.get()" (7개, dict literal 필수 포함)
  
- "try except 예외 처리하는 방법"
  → "try except exception handling built-in exceptions KeyError ValueError IndexError traceback" (8개, 구체 예외명 포함)
  
- "함수 정의하는 방법 def 키워드"
  → "function definition def keyword parameters arguments return statement callable" (7개, def keyword 필수 포함)
  
- "모듈 임포트 하는 방법"
  → "import statement from import module namespace standard library __init__.py package directory" (7개, import statement 필수 포함)
  
- "if elif else 조건문 사용법"
  → "if statement elif else conditional expression control flow boolean expression comparison operators" (7개, 쉼표 없이 공백으로만 구분)

Python 공식문서에서 실제로 사용되는 정확한 용어 (우선순위 높음):
- 연산자: floor division (//), modulo operator (%), power operator (**), arithmetic operators
- 자료구조: list.append(), list.extend(), list.insert(), list.remove(), list.pop(), list.clear(),
  list.index(), list.count(), list.sort(), list.reverse(), list.copy(),
  dict.get(), dict.keys(), dict.values(), dict.items(), dict.update(),
  dictionary display, dict literal, dict comprehension, list comprehension,
  tuple unpacking, set operations, sequence types, mapping types
- 제어문: if statement elif else for statement while statement break continue
  conditional expression match statement case statement
  (주의: 쉼표 없이 공백으로만 구분하여 사용)
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

최종 확인:
1. 키워드가 4개 이상인가? (2~3개면 관련 키워드 추가)
2. 핵심 개념의 필수 키워드가 포함되었는가?
3. 구체적인 함수/메서드/연산자 이름이 포함되었는가?
4. 금지 단어를 사용하지 않았는가?

한국어 질문: {query}
영어 키워드:"""