# Search Agent 프롬프트 수정 가이드

> 팀원이 프롬프트를 수정할 때 필요한 핵심 정보만 간단히 정리했습니다.

---

## 🔍 검색 시스템 구조 (한 줄 요약)

**질문 → Router가 "어디서 몇 개" 정함 → Search Agent가 실제 검색 → 결과 합쳐서 반환**

---

## 📁 프롬프트 파일 2개

### 1. `search_prompt.py` - 라우터 프롬프트

**역할**: 질문을 보고 **어디서 검색할지, 몇 개 가져올지** 결정

**결정하는 것**:
- `sources`: `['lecture']` (ML 강의) / `['python_doc']` (Python 문서) / 둘 다
- `top_k`: `3` (basic) / `5` (intermediate) / `7` (advanced)

**수정 예시**:
```python
# 키워드 추가: "xgboost"를 ML 키워드로 추가
- 머신러닝/딥러닝: rag, embedding, vector, 모델, 학습, 분류, 회귀, sklearn, iris, 결정트리, xgboost 등

# 효과: "xgboost 모델에 대해 설명해줘" → lecture에서 검색됨
```

---

### 2. `translate_prompt.py` - 번역 프롬프트

**역할**: 한글 질문을 **Python 공식문서에서 잘 검색되도록** 영어 키워드로 변환

**사용 조건**: 
- 질문이 한글이고
- Router가 `python_doc`를 검색 대상으로 선택했을 때만

**수정 예시**:
```python
# 키워드 예시 추가: "딕셔너리" 관련 더 구체적인 예시
예) list comprehension, dictionary display, dict literal, dict methods, dict.get(), dict.keys(), with statement

# 효과: "딕셔너리 사용법" → "dictionary display dict methods dict.get()" 같은 구체적 키워드로 변환
# → 검색 점수가 올라감
```

---

## 🔄 검색 로직 (간단 버전)

### 한글 질문일 때

1. **Router 호출** → `sources`, `top_k` 결정
2. **lecture 검색**: 한글 원문으로 검색
3. **python_doc_rst 검색**: 
   - 영어로 번역해서 검색 (기본)
   - 번역 결과가 약하면(top1 score < 0.45) 한글 원문으로도 1번 더 검색 (fallback)
4. **결과 합치기**: 중복 제거 → 점수순 정렬 → top_k개 반환

### 영어 질문일 때

1. **Router 호출** → `sources`, `top_k` 결정
2. **각 소스에서 원문(영어) 그대로 검색**
3. **결과 합치기**: 중복 제거 → 점수순 정렬 → top_k개 반환

---

## ✏️ 프롬프트 수정 체크리스트

### `search_prompt.py` 수정할 때

- ✅ **키워드 추가**: ML/Python 키워드 목록에 추가하면 해당 주제가 어느 소스로 분류되는지 바뀜
- ✅ **난이도 기준**: basic/intermediate/advanced 기준을 바꾸면 top_k 값이 바뀜
- ✅ **소스 판단 기준**: "이런 질문은 양쪽에서 검색" 같은 규칙 추가 가능

### `translate_prompt.py` 수정할 때

- ✅ **키워드 예시 추가**: 한글→영어 변환 시 어떤 키워드가 나오는지 바뀜
- ✅ **출력 형식**: "4~12개 키워드" 같은 제약 변경 가능
- ✅ **제약 강화**: 너무 일반적인 단어 금지 목록 추가 가능

---

## 🧪 테스트 방법

```bash
# Search Agent 테스트 실행
python src/agent/nodes/search_agent.py
```

**확인할 것**:
- Router가 올바른 `sources`를 선택하는지
- 번역 결과가 구체적인 키워드로 나오는지
- 검색 점수가 적절한지 (0.45 이상 권장)

---

## 📝 요약

- **`search_prompt.py`**: 어디서 검색할지, 몇 개 가져올지 결정
- **`translate_prompt.py`**: 한글 질문을 영어 키워드로 변환 (python_doc_rst 검색용)
- **검색 로직**: lecture는 원문 그대로, python_doc_rst는 번역 기본 + fallback

**프롬프트 수정 후 → 테스트 실행 → 결과 확인 → 필요하면 반복 개선**
