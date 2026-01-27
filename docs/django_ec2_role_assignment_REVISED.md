# Django 백엔드 & EC2 배포 역할 분담 (수정안)

> **4차 프로젝트** - Django 백엔드 & EC2 배포 팀 (주원, 지용, 경은) + UI 팀 (세연, 경은) | **3일 완성**

---

## 📋 전체 팀 구성

### UI 개발팀
- **세연**: 메인 UI 구조
- **경은**: 도움 UI + Django/EC2

### RAG 챗봇팀 (완성됨)
- **가람·혜빈**: 데이터 전처리(보강) & RAG 챗봇, 유사도 시각화 & 질문 필터링

### Django 백엔드 & 배포팀
- **주원·지용·경은**: Django 백엔드 & EC2 배포

---

## 👥 Django 팀 역할 분담 (수정안)

### 1️⃣ 세연 - **메인 UI 개발**
| 구분          | 내용                                  |
| ------------- | ------------------------------------- |
| **메인 역할** | Django 템플릿 메인 페이지 구현        |
| **작업 범위** | 기존 Flask `index.html` → Django 변환 |

**할 일:**
- [ ] `templates/base.html` 공통 레이아웃 작성
- [ ] `templates/main.html` 메인 채팅 UI 구현  
- [ ] 기존 2177줄 `index.html`의 CSS를 `static/css/main.css`로 분리
- [ ] JavaScript API 호출 코드를 Django URL(`{% url %}`)로 변경
- [ ] 주원님과 API 스펙 협의

---

### 2️⃣ 경은 - **도움 UI + 서브 백엔드 + 발표**
| 구분          | 내용                             |
| ------------- | -------------------------------- |
| **메인 역할** | 도움 페이지 UI + 백엔드 유틸리티 |
| **보조 역할** | 발표 자료 제작                   |

**할 일:**
- [ ] `templates/help.html` 도움말/가이드 페이지 구현
- [ ] Quiz API (`/api/quiz/`) Flask → Django 이관
- [ ] Health Check API (`/api/health/`) 구현
- [ ] 공통 에러 핸들링 미들웨어 구현 (`config/middleware.py`)
- [ ] API 테스트 코드 작성 (pytest)
- [ ] 발표 자료 제작

---

### 3️⃣ 주원 - **Django 백엔드 개발 리드**
| 구분          | 내용                                 |
| ------------- | ------------------------------------ |
| **메인 역할** | Django 프로젝트 구조 설계 & RAG 연동 |
| **리더 역할** | 코드 리뷰, 일정 관리, UI 팀 협업     |

**할 일:**
- [ ] Django 프로젝트 초기 세팅 (`django-admin startproject`)
- [ ] 앱 구조 설계 (`chat`, `quiz`, `common`)
- [ ] Chat API 엔드포인트 구현 (`/api/chat/`, `/api/chat/stream/`)
- [ ] Flask `app.py`의 `learning_agent()` 로직을 Django로 이관
- [ ] SSE 스트리밍 응답 구현 (기존 Flask 방식 참고)
- [ ] UI 팀(세연, 경은)과 API 스펙 공유 및 협의
- [ ] 전체 코드 리뷰 및 통합

---

### 4️⃣ 지용 - **EC2 배포 & 인프라**
| 구분          | 내용                     |
| ------------- | ------------------------ |
| **메인 역할** | AWS EC2 배포 & 서버 운영 |
| **보조 역할** | Docker 컨테이너화        |

**할 일:**
- [x] Dockerfile 작성 (Django + Gunicorn)
- [x] docker-compose.yml 작성 (Django + Qdrant + Nginx)
- [x] nginx.conf 작성
- [x] .env.example 템플릿 작성
- [x] EC2 배포 가이드 문서화
- [ ] Django `collectstatic` 설정 검토
- [ ] AWS EC2 인스턴스 생성 및 설정
- [ ] Nginx + Gunicorn 세팅
- [ ] Qdrant 서버 EC2 배포
- [ ] 환경 변수 관리 (`.env`)
- [ ] 배포 테스트 및 안정화

---

## 🔧 Django 프로젝트 구조 (수정안)

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
│   │   ├── quiz/          # 퀴즈 API (경은)
│   │   │   ├── views.py
│   │   │   ├── urls.py
│   │   │   └── services.py
│   │   │
│   │   └── common/        # 공통 기능 (경은)
│   │       ├── views.py   # Health Check, 메인/도움 페이지
│   │       └── urls.py
│   │
│   ├── templates/         # 🎨 UI 템플릿
│   │   ├── base.html      # 공통 레이아웃 (주원)
│   │   ├── main.html      # 메인 채팅 UI (주원)
│   │   └── help.html      # 도움말 페이지 (경은)
│   │
│   ├── static/            # 정적 파일 (경은 + 지용)
│   │   ├── css/
│   │   │   ├── base.css   # 공통 스타일 (경은)
│   │   │   ├── main.css   # 메인 페이지 (경은)
│   │   │   └── help.css   # 도움 페이지 (경은)
│   │   ├── js/
│   │   │   ├── chat.js    # 채팅 로직 (경은)
│   │   │   └── quiz.js    # 퀴즈 로직 (경은)
│   │   └── images/        # (기존 image/ 폴더 복사)
│   │
│   └── tests/             # 테스트 코드 (경은)
│       ├── test_chat.py
│       └── test_quiz.py
│
├── main.py                # ✅ RAG 시스템 (가람·혜빈, 그대로 유지)
├── src/                   # ✅ RAG 코어 로직 (가람·혜빈, 그대로 유지)
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

### 핵심 서비스 (`apps/chat/services.py`) - 주원

```python
import sys
from pathlib import Path

# 기존 프로젝트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import main

class RAGChatService:
    """Flask app.py의 learning_agent() 로직을 Django로 이관"""
    
    def get_answer(self, question: str):
        # main() 함수 호출 (기존 RAG 시스템)
        result = main(question)
        
        # Flask app.py와 동일한 응답 포맷
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
        
        return {
            'text': answer_text,
            'sources': sources,
            'suggested_questions': result.get('suggested_questions', []),
            'steps': [
                {'step': 1, 'title': 'Router', 'desc': '질문 유형 분석'},
                {'step': 2, 'title': 'Search', 'desc': f'Qdrant 검색 ({len(search_results)}개)'},
                {'step': 3, 'title': 'Analyst', 'desc': '답변 생성 완료'}
            ]
        }
```

### Health Check API (`apps/common/views.py`) - 경은

```python
from django.http import JsonResponse
from qdrant_client import QdrantClient
import os

def health_check(request):
    """Qdrant 연결 상태 확인"""
    try:
        client = QdrantClient(
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', 6333))
        )
        client.get_collections()
        return JsonResponse({'status': 'ok', 'db': 'connected'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=503)

def main_page(request):
    """메인 채팅 페이지 렌더링 (세연 UI)"""
    return render(request, 'main.html')

def help_page(request):
    """도움말 페이지 렌더링 (경은 UI)"""
    return render(request, 'help.html')
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
- `templates/index.html` (2177줄) → `main.html` + `help.html` + CSS 분리
- API 엔드포인트 경로 동일하게 유지 (프론트엔드 최소 수정)

### 3. UI 팀-백엔드 팀 협업
- **세연 ↔ 주원**: 메인 페이지 API 스펙 협의
- **경은 ↔ 주원**: 도움 페이지 + Quiz API 협의
- JSON 응답 형식은 기존 Flask와 동일하게 유지

---

## 📞 협업 규칙

### Git 브랜치 전략
- `feature/django-init` - Django 프로젝트 초기화 (주원)
- `feature/ui-main` - 메인 UI 구현 (세연)
- `feature/ui-help` - 도움 UI 구현 (경은)
- `feature/backend-chat` - Chat API (주원)
- `feature/backend-quiz` - Quiz API (경은)
- `feature/ec2-deploy` - EC2 배포 (지용)

### 커밋 메시지 규칙
- `[FE]` Frontend (UI)
- `[BE]` Backend (API)
- `[DEPLOY]` Deployment
- `[TEST]` Testing
- `[DOCS]` Documentation

### PR 리뷰 규칙
- 머지 전 최소 1명 이상 리뷰
- UI 변경 시 스크린샷 첨부
- API 변경 시 Postman 테스트 결과 공유

---

## 📅 3일 일정 (예시)

### Day 1: 프로젝트 설정 & 구조 잡기
- **주원**: Django 프로젝트 생성, 앱 구조 설계, `base.html` + `main.html` 템플릿 작성 시작
- **경은**: 기존 `index.html` CSS/JS 분석 및 분리 계획, 도움 페이지 설계
- **지용**: Dockerfile 검토, 로컬 Qdrant 테스트, Django static 설정 준비
- **세연**: UI 디자인 방향성 및 아이디어 공유 (선택사항)

### Day 2: 개발
- **주원**: `main.html` 완성 + Chat API 구현 + SSE 스트리밍
- **경은**: CSS 3개 파일 분리, JS 파일 분리, `help.html` 구현, Quiz/Health API 구현
- **지용**: EC2 인스턴스 생성, 보안 그룹 설정, collectstatic 자동화

### Day 3: 통합 & 배포
- **오전**: 로컬 통합 테스트 (전체)
- **오후**: EC2 배포 (지용), Nginx 설정, 최종 검증 (전체)
- **저녁**: 발표 자료 완성 (경은)

---

## 🎯 핵심 변경 사항 요약

### 기존 역할 분담과의 차이점

| 항목           | 기존         | 수정안                           |
| -------------- | ------------ | -------------------------------- |
| **UI 개발**    | 명시 없음    | 세연(메인), 경은(도움) 명시      |
| **프론트엔드** | 별도 팀 가정 | Django 템플릿으로 통합           |
| **Flask 처리** | 언급 없음    | Flask → Django 마이그레이션 명시 |
| **경은 역할**  | Quiz API만   | Quiz API + 도움 UI + 발표        |
| **주원 역할**  | 백엔드만     | 백엔드 + UI 팀 협업              |
| **지용 역할**  | 변경 없음    | ✅ (이미 준비 완료)               |

---

> **마지막 업데이트:** 2026-01-27  
> **작성자:** 지용 (EC2 배포 & 인프라 담당)
