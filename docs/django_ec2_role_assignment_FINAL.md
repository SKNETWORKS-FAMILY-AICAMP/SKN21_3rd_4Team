# Django 백엔드 & EC2 배포 역할 분담 (최종안)

> **4차 프로젝트** - Django 백엔드 & EC2 배포 팀 (주원, 지용, 경은) | **3일 완성**

---

## 📋 전체 팀 구성

### UI 디자인 (컨설팅)
- **세연**: 메인 UI 디자인 아이디어 제공 (**개발 X**, 디자인 방향성만)

### RAG 챗봇팀 (완성됨)
- **가람·혜빈**: 데이터 전처리(보강) & RAG 챗봇, 유사도 시각화 & 질문 필터링

### Django 백엔드 & UI 개발 & 배포팀
- **주원·경은·지용**: Django 백엔드 + 프론트엔드 + EC2 배포 **전부 담당**

---

## 👥 Django 팀 역할 분담 (최종안)

### 1️⃣ 주원 - **Django 백엔드 리드 + 메인 UI 구현**
| 구분          | 내용                                 |
| ------------- | ------------------------------------ |
| **메인 역할** | Django 프로젝트 전체 설계 & RAG 연동 |
| **UI 역할**   | 메인 채팅 페이지 템플릿 구현         |
| **리더 역할** | 코드 리뷰, 일정 관리                 |

**할 일:**
- [ ] Django 프로젝트 초기 세팅 (`django-admin startproject`)
- [ ] 앱 구조 설계 (`chat`, `quiz`)
- [ ] **`templates/base.html` 공통 레이아웃 작성**
- [ ] **`templates/main.html` 메인 채팅 UI 구현** (세연 디자인 아이디어 참고)
- [ ] **Flask `index.html` (2177줄) → Django 템플릿 변환**
- [ ] Chat API 엔드포인트 구현 (`/api/chat/`, `/api/chat/stream/`)
- [ ] Flask `app.py`의 `learning_agent()` 로직을 Django로 이관
- [ ] SSE 스트리밍 응답 구현
- [ ] 전체 코드 리뷰 및 통합

---

### 2️⃣ 경은 - **Quiz API + CSS/JS**
| 구분          | 내용                |
| ------------- | ------------------- |
| **메인 역할** | Quiz API + 스타일링 |

**할 일:**
- [ ] **기존 `index.html`의 CSS를 `static/css/` 폴더로 분리**
  - `static/css/base.css` (공통 스타일)
  - `static/css/main.css` (메인 페이지 스타일)
- [ ] **JavaScript 파일 분리 및 Django URL 연동**
  - `static/js/chat.js` (채팅 로직)
  - `static/js/quiz.js` (퀴즈 로직)
- [ ] Quiz API (`/api/quiz/`) Flask → Django 이관
- [ ] 공통 에러 핸들링 미들웨어 (`config/middleware.py`)

---

### 3️⃣ 지용 - **EC2 배포 & 인프라**
| 구분          | 내용                     |
| ------------- | ------------------------ |
| **메인 역할** | AWS EC2 배포 & 서버 운영 |
| **보조 역할** | Django 정적 파일 설정    |

**할 일:**
- [x] Dockerfile 작성 (Django + Gunicorn)
- [x] docker-compose.yml 작성
- [x] nginx.conf 작성
- [x] .env.example 템플릿 작성
- [x] EC2 배포 가이드 문서화
- [ ] Django `settings.py`에서 `STATIC_ROOT`, `STATICFILES_DIRS` 설정
- [ ] `python manage.py collectstatic` 명령 자동화
- [ ] Nginx가 `/static/` 경로를 직접 서빙하도록 설정
- [ ] AWS EC2 인스턴스 생성 및 설정
- [ ] Nginx + Gunicorn 세팅
- [ ] Qdrant 서버 EC2 배포
- [ ] 환경 변수 관리 (`.env`)
- [ ] 배포 테스트 및 안정화

---

## 🔧 Django 프로젝트 구조 (최종안)

```
SKN21_3rd_4Team/
├── django_app/            # 🆕 새로 만들 Django 프로젝트
│   ├── manage.py
│   │
│   ├── config/            # Django 설정 (주원)
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
│   │   └── middleware.py  # 에러 핸들링 (경은)
│   │
│   ├── apps/
│   │   ├── chat/          # 챗봇 API (주원)
│   │   │   ├── views.py
│   │   │   ├── urls.py
│   │   │   └── services.py  # RAG 연동 (main() 호출)
│   │   │
│   │   └── quiz/          # 퀴즈 API (경은)
│   │       ├── views.py
│   │       ├── urls.py
│   │       └── services.py
│   │
│   ├── templates/         # 🎨 UI 템플릿
│   │   ├── base.html      # 공통 레이아웃 (주원)
│   │   └── main.html      # 메인 채팅 UI (주원)
│   │
│   ├── static/            # 정적 파일 (경은: CSS/JS, 지용: 설정)
│   │   ├── css/
│   │   │   ├── base.css   # 공통 스타일 (경은)
│   │   │   └── main.css   # 메인 페이지 (경은)
│   │   ├── js/
│   │   │   ├── chat.js    # 채팅 로직 (경은)
│   │   │   └── quiz.js    # 퀴즈 로직 (경은)
│   │   └── images/
│   │
│   └── tests/             # 테스트 코드 (선택)
│       └── test_api.py
│
├── main.py                # ✅ RAG 시스템 (가람·혜빈, 건드리지 말 것)
├── src/                   # ✅ RAG 코어 로직 (가람·혜빈, 건드리지 말 것)
├── data/                  # ✅ 강의 자료 (완성)
│
├── deploy/                # ✅ 배포 파일 (지용, 완료)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── .env.example
│
├── app.py                 # ❌ Flask 앱 (Django로 대체 예정)
└── templates/
    └── index.html         # ❌ 2177줄 파일 (Django로 분할 예정)
```

---

## 🤖 RAG 챗봇 Django 연동

### Chat API (`apps/chat/views.py`) - 주원

```python
from django.http import StreamingHttpResponse
import sys
from pathlib import Path
import json
import time

# 기존 프로젝트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import main

def chat_stream(request):
    """SSE 스트리밍 Chat API"""
    message = request.POST.get('message')
    
    def event_stream():
        # main() 호출 (RAG 시스템)
        result = main(message)
        
        # Flask app.py와 동일한 로직
        analyst_result = result.get('analyst_results', [])
        if analyst_result:
            last_msg = analyst_result[-1]
            answer_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        else:
            answer_text = "답변을 생성할 수 없습니다."
        
        # 검색 결과 처리
        search_results = result.get('search_results', [])
        sources = []
        for r in search_results[:3]:
            if r.get('score', 0) > 0.5:
                sources.append({
                    'type': r.get('metadata', {}).get('source', 'IPYNB').upper(),
                    'title': r.get('metadata', {}).get('lecture_title', '문서'),
                    'score': r.get('score', 0),
                    'content': r.get('content', '')[:200]
                })
        
        # 1단계: 진행 단계
        steps = [
            {'step': 1, 'title': 'Router', 'desc': '질문 분석'},
            {'step': 2, 'title': 'Search', 'desc': f'{len(search_results)}개 문서 검색'},
            {'step': 3, 'title': 'Analyst', 'desc': '답변 생성'}
        ]
        for step in steps:
            yield f"data: {json.dumps({'type': 'step', 'data': step})}\n\n"
            time.sleep(0.5)
        
        # 2단계: 답변 스트리밍
        for char in answer_text:
            yield f"data: {json.dumps({'type': 'char', 'data': char})}\n\n"
            time.sleep(0.02)
        
        # 3단계: 추천 질문
        suggested = result.get('suggested_questions', [])
        if suggested:
            yield f"data: {json.dumps({'type': 'suggestions', 'data': suggested})}\n\n"
        
        # 4단계: 참고 자료
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        
        # 완료
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
```

### Quiz API (`apps/quiz/views.py`) - 경은

```python
from django.http import JsonResponse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.quiz_service import QuizService

quiz_service = QuizService()

def get_quiz(request):
    """퀴즈 데이터 반환 API"""
    category = request.GET.get('category', 'all')
    count = int(request.GET.get('count', 5))
    quizzes = quiz_service.get_quizzes(category, count)
    return JsonResponse({'success': True, 'quizzes': quizzes})
```

### 에러 핸들링 미들웨어 (`config/middleware.py`) - 경은

```python
from django.http import JsonResponse

class ErrorHandlerMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        return JsonResponse({
            'success': False,
            'error': str(exception),
            'type': type(exception).__name__
        }, status=500)
```

---

## ⚠️ 작업 시 고려 사항

### 1. RAG 팀과 충돌 방지
- `main(question)` 함수만 호출, **내부 수정 절대 금지**
- `src/` 폴더는 **RAG 팀 전용** (가람·혜빈)
- 반환 형식 변경 시 사전 공유 필수

### 2. Flask → Django 마이그레이션
- 기존 `app.py`의 `/chat/stream` 로직을 `apps/chat/views.py`로 이관
- `templates/index.html` (2177줄) → `base.html` + `main.html`로 분할
- API 엔드포인트 경로 동일하게 유지

### 3. 간소화 전략
- ❌ 도움 페이지 제외 (시간 절약)
- ❌ Health Check API 제외 (필수 아님)
- ✅ **핵심 기능에만 집중** (Chat + Quiz)

---

## 🎯 핵심 변경 사항

| 항목             | 기존 문서      | 최종안 (간소화)                  |
| ---------------- | -------------- | -------------------------------- |
| **세연 역할**    | 메인 UI 개발   | ❌ **디자인 아이디어만** (개발 X) |
| **도움 UI**      | help.html 포함 | ❌ **제거** (불필요)              |
| **Health Check** | API 포함       | ❌ **제거** (선택사항)            |
| **주원 역할**    | 백엔드만       | ✅ 백엔드 + 메인 UI 템플릿        |
| **경은 역할**    | 많은 작업      | ✅ Quiz API + **CSS/JS만**        |
| **지용 역할**    | 배포만         | ✅ 배포 + 정적 파일 설정          |
| **템플릿 개수**  | 3개            | ✅ **2개** (base.html, main.html) |

---


---

> **마지막 업데이트:** 2026-01-27  
> **작성자:** 지용 (EC2 배포 & 인프라 담당)  
> **버전:** v3 (간소화 - 도움 UI & Health Check 제거)
