# Django 백엔드 & EC2 배포 역할 분담

> **4차 프로젝트** - Django 백엔드 & EC2 배포 팀 (주원, 지용, 경은) | **3일 완성**

---

## 👥 팀원별 역할 분담

### 1️⃣ 경은 - **발표자 & 서브 백엔드**
| 구분          | 내용                     |
| ------------- | ------------------------ |
| **메인 역할** | 발표 준비                |
| **개발 역할** | 퀴즈 API + 유틸리티 기능 |

**할 일:**
- [ ] 발표 자료 제작
- [ ] 퀴즈 API (`/api/quiz/`) Django 이관 및 구현
- [ ] Health Check API (`/api/health/`) 구현
- [ ] 공통 에러 핸들링 미들웨어 구현
- [ ] API 테스트 코드 작성 (pytest)

---

### 2️⃣ 주원 - **Django 백엔드 개발 리드**
| 구분          | 내용                                 |
| ------------- | ------------------------------------ |
| **메인 역할** | Django 프로젝트 구조 설계 & RAG 연동 |
| **리더 역할** | 코드 리뷰, 일정 관리                 |

**할 일:**
- [ ] Django 프로젝트 초기 세팅
- [ ] 앱 구조 설계 (`chat`, `quiz`, `common`)
- [ ] Chat API 엔드포인트 구현 (`/api/chat/`, `/api/chat/stream/`)
- [ ] RAG 워크플로우 연동 (LangGraph → Django)
- [ ] 스트리밍 응답 구현 (SSE)
- [ ] 코드 리뷰 및 통합

---

### 3️⃣ 지용 - **EC2 배포 & 인프라**
| 구분          | 내용                     |
| ------------- | ------------------------ |
| **메인 역할** | AWS EC2 배포 & 서버 운영 |
| **보조 역할** | Docker 컨테이너화        |

**할 일:**
- [ ] AWS EC2 인스턴스 생성 및 설정
- [ ] Nginx + Gunicorn 세팅
- [ ] Docker & Docker Compose 설정
- [ ] Qdrant 서버 EC2 배포
- [ ] 환경 변수 관리 (`.env`)
- [ ] 배포 테스트 및 안정화

---

## 🔧 Django 프로젝트 구조

```
django_app/
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── middleware.py      # 에러 핸들링 (경은)
│
├── apps/
│   ├── chat/              # 챗봇 API (주원)
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── services.py    # RAG 연동
│   │
│   ├── quiz/              # 퀴즈 API (경은)
│   │   ├── views.py
│   │   └── services.py
│   │
│   └── common/            # 공통 기능 (경은)
│       └── views.py       # Health Check
│
├── tests/                 # 테스트 코드 (경은)
│   ├── test_chat.py
│   └── test_quiz.py
│
├── Dockerfile             # (지용)
└── docker-compose.yml     # (지용)
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
    def get_answer(self, question: str):
        result = main(question)
        return {
            'answer': result.get('analyst_results', [{}])[-1].content,
            'sources': result.get('search_results', [])[:3],
            'suggestions': result.get('suggested_questions', [])
        }
```

### Health Check API (`apps/common/views.py`) - 경은

```python
from django.http import JsonResponse
from qdrant_client import QdrantClient
import os

def health_check(request):
    try:
        client = QdrantClient(
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', 6333))
        )
        client.get_collections()
        return JsonResponse({'status': 'ok', 'db': 'connected'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=503)
```

### 에러 핸들링 미들웨어 (`config/middleware.py`) - 경은

```python
from django.http import JsonResponse
import traceback

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

### 챗봇 팀과 충돌 방지
- `main(question)` 함수만 호출, 내부 수정 X
- `src/` 폴더 = 챗봇 팀 전용
- 반환 형식 변경 시 사전 공유

### API 스펙 (UI 팀 공유용)
```json
// POST /api/chat/
{"message": "RAG가 뭐야?"}

// Response
{"success": true, "answer": "...", "sources": [...]}

// GET /api/health/
{"status": "ok", "db": "connected"}

// GET /api/quiz/?category=python&count=5
{"success": true, "quizzes": [...]}
```

### EC2 체크리스트
- [ ] 인스턴스: t3.medium 이상
- [ ] 포트: 80, 443, 22, 6333 오픈
- [ ] Swap 메모리 2GB 설정

---

## 📞 협업 규칙

- **브랜치**: `feature/django-chat`, `feature/django-quiz`, `feature/ec2-deploy`
- **커밋**: `[BE]`, `[DEPLOY]`, `[TEST]` 접두사
- **PR**: 머지 전 1명 이상 리뷰

---

> 마지막 업데이트: 2026-01-27
