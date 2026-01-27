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
- [ ] 앱 구조 설계 (`chat`, `quiz`, `common`)
- [ ] **`templates/base.html` 공통 레이아웃 작성**
- [ ] **`templates/main.html` 메인 채팅 UI 구현** (세연 디자인 아이디어 참고)
- [ ] **Flask `index.html` (2177줄) → Django 템플릿 변환**
- [ ] Chat API 엔드포인트 구현 (`/api/chat/`, `/api/chat/stream/`)
- [ ] Flask `app.py`의 `learning_agent()` 로직을 Django로 이관
- [ ] SSE 스트리밍 응답 구현
- [ ] 전체 코드 리뷰 및 통합

---

### 2️⃣ 경은 - **도움 UI + Quiz API + CSS/JS + 발표**
| 구분          | 내용                                 |
| ------------- | ------------------------------------ |
| **메인 역할** | 도움 페이지 UI + Quiz API + 스타일링 |
| **보조 역할** | 발표 자료 제작                       |

**할 일:**
- [ ] **`templates/help.html` 도움말/가이드 페이지 구현**
- [ ] **기존 `index.html`의 CSS를 `static/css/` 폴더로 분리**
  - `static/css/base.css` (공통 스타일)
  - `static/css/main.css` (메인 페이지 스타일)
  - `static/css/help.css` (도움 페이지 스타일)
- [ ] **JavaScript 파일 분리 및 Django URL 연동**
  - `static/js/chat.js` (채팅 로직)
  - `static/js/quiz.js` (퀴즈 로직)
- [ ] Quiz API (`/api/quiz/`) Flask → Django 이관
- [ ] Health Check API (`/api/health/`) 구현
- [ ] 공통 에러 핸들링 미들웨어 (`config/middleware.py`)
- [ ] API 테스트 코드 작성 (pytest)
- [ ] 발표 자료 제작

---

### 3️⃣ 지용 - **EC2 배포 & 인프라 + 정적 파일 관리**
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
- [ ] **Django `settings.py`에서 `STATIC_ROOT`, `STATICFILES_DIRS` 설정**
- [ ] **`python manage.py collectstatic` 명령 자동화**
- [ ] **Nginx가 `/static/` 경로를 직접 서빙하도록 설정**
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
│   │   ├── main.html      # 메인 채팅 UI (주원 - 세연 디자인 참고)
│   │   └── help.html      # 도움말 페이지 (경은)
│   │
│   ├── static/            # 정적 파일 (경은: CSS/JS, 지용: 설정)
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


## 🎯 핵심 변경 사항

| 항목          | 기존 문서    | 최종안                                        |
| ------------- | ------------ | --------------------------------------------- |
| **세연 역할** | 메인 UI 개발 | ❌ **디자인 아이디어만 제공** (개발 X)         |
| **UI 개발**   | 별도 팀      | ✅ **Django 팀이 전부 담당**                   |
| **주원 역할** | 백엔드만     | ✅ 백엔드 + **메인 UI 템플릿**                 |
| **경은 역할** | Quiz API만   | ✅ Quiz API + 도움 UI + **CSS/JS 전체** + 발표 |
| **지용 역할** | 배포만       | ✅ 배포 + **정적 파일 설정**                   |

---

> **마지막 업데이트:** 2026-01-27  
> **작성자:** 지용 (EC2 배포 & 인프라 담당)
