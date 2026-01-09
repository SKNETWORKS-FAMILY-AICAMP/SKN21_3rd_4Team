# 테스트 질문별 추천 RST 파일

## 현재 테스트 질문 (11개)

### 1. 숫자 연산 / 정수 나눗셈
- **질문**: "파이썬에서 숫자 연산하는 방법", "정수 나눗셈과 나머지 연산자 사용법"
- **추천 파일**:
  - `tutorial/introduction.rst` (기초 연산)
  - `reference/expressions.rst` (표현식, 연산자 상세)

### 2. 문자열 슬라이싱
- **질문**: "문자열 슬라이싱 하는 법"
- **추천 파일**:
  - `tutorial/introduction.rst` (기초)
  - `reference/expressions.rst` (Slicings 섹션)

### 3. 리스트 컴프리헨션
- **질문**: "리스트 컴프리헨션이란"
- **추천 파일**:
  - `tutorial/datastructures.rst` (List Comprehensions)
  - `reference/expressions.rst` (Generator expressions)

### 4. if elif else
- **질문**: "if elif else 조건문 사용법"
- **추천 파일**:
  - `tutorial/controlflow.rst` (if Statements)
  - `reference/compound_stmts.rst` (The if statement)

### 5. for문 range
- **질문**: "for문에서 range 함수 사용하는 방법"
- **추천 파일**:
  - `tutorial/controlflow.rst` (The range Function)

### 6. 함수 정의
- **질문**: "함수 정의하는 방법 def 키워드"
- **추천 파일**:
  - `tutorial/controlflow.rst` (Defining Functions)
  - `reference/compound_stmts.rst` (Function definitions)

### 7. 람다 함수
- **질문**: "람다 함수 사용법"
- **추천 파일**:
  - `tutorial/controlflow.rst` (Lambda Expressions)
  - `reference/expressions.rst` (Lambdas)

### 8. 딕셔너리 리터럴
- **질문**: "딕셔너리 리터럴 사용법"
- **추천 파일**:
  - `tutorial/datastructures.rst` (Dictionaries)
  - `reference/expressions.rst` (Dictionary displays)

### 9. 모듈 임포트
- **질문**: "모듈 임포트 하는 방법"
- **추천 파일**:
  - `tutorial/modules.rst` (Modules)
  - `reference/import.rst` (The import statement)

### 10. try except
- **질문**: "try except 예외 처리하는 방법"
- **추천 파일**:
  - `tutorial/errors.rst` (Errors and Exceptions)
  - `reference/compound_stmts.rst` (The try statement)

---

## 최소 테스트 세트 (빠른 테스트용)

**핵심 파일 6개만**:
1. `tutorial/introduction.rst` - 기초 (숫자, 문자열)
2. `tutorial/controlflow.rst` - 제어문, 함수, 람다
3. `tutorial/datastructures.rst` - 리스트, 딕셔너리
4. `tutorial/modules.rst` - 모듈
5. `tutorial/errors.rst` - 예외
6. `reference/expressions.rst` - 표현식 (연산자, 컴프리헨션, 슬라이싱)

**또는 reference만 (더 상세)**:
1. `reference/expressions.rst` - 대부분의 질문 커버
2. `reference/compound_stmts.rst` - if, for, def, try
3. `reference/import.rst` - import

---

## 사용 방법

### 방법 1: 특정 파일만 선택
```bash
# tutorial의 핵심 파일만
python src/ingestion_rst.py --subdirs tutorial --recreate-collection --collection learning_ai_test
```

### 방법 2: 특정 파일 리스트로 직접 지정 (수정 필요)
현재는 디렉토리 단위만 지원하므로, 가장 작은 세트는:
```bash
# tutorial만 (17개 파일, 약 1-2분)
python src/ingestion_rst.py --subdirs tutorial --recreate-collection --collection learning_ai_test
```

### 방법 3: reference만 (더 상세, 11개 파일)
```bash
python src/ingestion_rst.py --subdirs reference --recreate-collection --collection learning_ai_test
```
