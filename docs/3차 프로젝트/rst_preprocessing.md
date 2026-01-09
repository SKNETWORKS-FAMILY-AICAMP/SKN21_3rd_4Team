# RST/Python Docs 전처리 로직 설명

본 문서는 `src/ingestion_rst_test.py`에 구현된 Python 공식 문서(RST) 전처리 및 Ingestion 로직에 대해 설명합니다.  
RAG(Retrieval-Augmented Generation) 시스템의 검색 품질을 높이기 위해 원본 RST 파일의 구조적 특성을 고려하여 노이즈를 제거하고 의미 단위로 청킹합니다.

---

## 1. 개요 (Overview)

- **대상 파일**: Python 공식 문서 소스 (`.rst` 파일)
- **목적**: 불필요한 문법적 노이즈(Directive, 장식 문자 등)를 제거하고, 문서의 계층 구조(Section)를 보존하여 벡터 검색에 최적화된 텍스트 청크를 생성합니다.
- **담당 클래스**: `RSTIngestor`

---

## 2. 주요 전처리 단계 (Preprocessing Steps)

전체 과정은 **파일 파싱 → 섹션 분리 → 노이즈 정제 → 청킹** 순으로 진행됩니다.

### 2.1 섹션 파싱 (`_parse_rst_sections`)
RST 파일은 들여쓰기와 밑줄(underline)로 구조가 잡혀 있습니다. 이를 활용해 문서를 계층적인 섹션 단위로 1차 분리합니다.

- **H1 (최상위 제목)**: `=====` 밑줄을 사용한 제목
- **H2 (부제목)**: `-----` 밑줄을 사용한 제목
- **그 외**: H3 이하의 소제목은 별도 섹션으로 분리하지 않고 현재 섹션의 본문에 포함시킵니다.
- **결과**: `(H1 제목, H2 제목, 본문 내용)` 형태의 튜플 리스트 생성

### 2.2 노이즈 정제 (`_clean_rst_noise`)
검색 정확도에 악영향을 줄 수 있는 RST 특유의 문법 요소를 제거하거나 일반 텍스트로 변환합니다.

1.  **Directive 제거**:
    - **스킵 대상**: 메타데이터 성격의 지시어 (`.. highlight::`, `.. versionadded::`, `.. seealso::` 등)
    - **블록 제거**: 목차나 인덱스 (`.. toctree::`, `.. index::`) 등 검색 가치가 없는 블록은 통째로 제외

2.  **코드 블록 처리**:
    - `.. code-block:: python`과 같은 구문은 내용(**코드 본문**)을 유지합니다.
    - 언어에 대한 힌트(예: `python`)가 있다면 텍스트로 남겨 검색 맥락을 보존합니다.

3.  **API 시그니처 보존**:
    - `.. py:class::`, `.. py:function::` 등 라이브러리 정의 구문은 directive 문법만 떼어내고, **함수/클래스 정의부(Signature)** 텍스트는 그대로 살립니다. (예: `class PurePath(*pathsegments)`)

4.  **RST Role 및 링크 정리**:
    - `:role:\`content\`` 형태를 `content`로 치환 (예: `:func:\`open\`` → `open`)
    - 링크 구문 (`\`text\`_`) 제거하고 텍스트만 유지
    - 꺾쇠 괄호로 된 링크(`<...>`) 제거

5.  **장식 제거**:
    - 제목을 꾸미기 위한 특수문자 라인(`===`, `---` 등) 제거

### 2.3 인라인 마크업 정리 (`_clean_inline_markup`)
제목이나 헤더에 포함된 인라인 마크업을 정리하여 깔끔한 제목 텍스트를 추출합니다.
- `:role:\`...\`` → `...`
- `\`text\`_` → `text`

---

## 3. 청킹 전략 (Chunking Strategy)

정제된 텍스트를 LLM이 이해하기 좋은 크기로 자르는(Chunking) 과정입니다.

### 3.1 Recursive Splitting
`RecursiveCharacterTextSplitter`를 사용하여 의미론적 단위가 깨지지 않도록 자릅니다.
- **Chunk Size**: 900자 (토큰 아님, 글자 수 기준)
- **Overlap**: 200자 (문맥 단절 방지)
- **구분자 우선순위**:
    1. `\n\n` (문단 바꿈)
    2. `\n::` (코드 블록 시작)
    3. `\n.. ` (Directive 시작)
    4. `\n` (줄바꿈)

### 3.2 문맥 주입 (Context Injection)
각 청크가 문서의 어느 부분인지 알 수 있도록 **메타데이터를 텍스트 앞단에 명시**합니다. 검색된 청크만 봤을 때도 전체 맥락을 파악할 수 있게 돕습니다.

**청크 구조 예시**:
```text
[TITLE] introduction
[H1] Python 소개
[H2] 주요 특징

(여기서부터 실제 청크 내용...)
Python은 배우기 쉽고 강력한 프로그래밍 언어입니다...
```

---

## 4. Ingestion 결과물 (Vector DB)

위 과정을 거친 데이터는 Qdrant 벡터 DB에 다음과 같은 메타데이터와 함께 저장됩니다.

- `source`: `python_doc_rst`
- `title`: 파일명 (e.g., `introduction`)
- `section`: H1 제목
- `subsection`: H2 제목 (없으면 H1과 동일)
- `has_code`: 코드 블록 포함 여부 (True/False)
- `snippet`: 청크의 앞부분 미리보기 (디버깅용)
- `chunk_index`: 원본내 순서
