# Flask Backend for Bootcamp AI Tutor
# 기능: 학습 에이전트, 스트리밍 응답
# ============================================

# [Flask 핵심 모듈 임포트]
from flask import Flask, render_template, request, jsonify, session, Response
import json
import os
import time
import uuid

# [Flask 앱 인스턴스 생성]
app = Flask(__name__)

# [비밀 키 설정]
app.secret_key = 'bootcamp-ai-tutor-secret-key-2024'

# ============================================
# 설정 (Configuration)
# ============================================

# [사용 가능한 모드(에이전트) 정의]
MODES = {
    'learning': {'name': '학습할래용', 'icon': '📚', 'system_prompt': '친절한 학습 튜터로서 답변해주세요.'},
}


# ============================================
# Agent Functions (Mode-specific logic)
# ============================================


# ============================================
USE_REAL_BACKEND = False  # True로 변경하면 실제 백엔드 사용
BACKEND_URL = "http://localhost:8000"  # 백엔드 URL

# [학습 자료 기반 샘플 응답 데이터]
# 실제 ipynb 파일 목록 기반
SAMPLE_RESPONSES = {
    '머신러닝': {
        'text': """## 🤖 머신러닝 개요

**머신러닝(Machine Learning)**은 컴퓨터가 명시적인 프로그래밍 없이 데이터로부터 학습하여 패턴을 찾고 예측하는 인공지능의 한 분야입니다.

### 주요 개념
1. **지도학습 (Supervised Learning)**: 정답이 있는 데이터로 학습
   - 분류(Classification): 카테고리 예측
   - 회귀(Regression): 연속값 예측

2. **비지도학습 (Unsupervised Learning)**: 정답 없이 패턴 발견
   - 군집화(Clustering): 유사한 데이터 그룹화
   - 차원 축소: 데이터 압축

3. **강화학습 (Reinforcement Learning)**: 보상을 통한 학습

### 학습 과정
```python
# 기본 머신러닝 워크플로우
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. 전처리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. 모델 학습
model.fit(X_train_scaled, y_train)

# 4. 예측 및 평가
predictions = model.predict(X_test_scaled)
```""",
        'sources': [
            {'type': 'IPYNB', 'title': '01_머신러닝개요.ipynb', 'content': '머신러닝 기초 개념'},
            {'type': 'IPYNB', 'title': '02_첫번째 머신러닝 분석 - Iris_분석.ipynb', 'content': 'Iris 데이터셋 실습'}
        ]
    },
    '전처리': {
        'text': """## 📊 데이터 전처리

데이터 전처리는 머신러닝 모델의 성능을 좌우하는 핵심 단계입니다.

### 주요 전처리 기법

1. **결측치 처리**
```python
# 결측치 확인
df.isnull().sum()

# 평균값으로 대체
df.fillna(df.mean(), inplace=True)

# 행 삭제
df.dropna(inplace=True)
```

2. **스케일링 (정규화)**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 표준화 (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max 정규화 (0~1)
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X)
```

3. **인코딩**
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 라벨 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 원-핫 인코딩
df_encoded = pd.get_dummies(df, columns=['category'])
```

4. **이상치 처리**
- IQR 방법으로 이상치 탐지
- Z-score 기반 제거""",
        'sources': [
            {'type': 'IPYNB', 'title': '04_데이터_전처리.ipynb', 'content': '전처리 기법 상세'},
            {'type': 'IPYNB', 'title': '03_데이터셋 나누기와 모델검증.ipynb', 'content': 'Train/Test 분리'}
        ]
    },
    '결정트리': {
        'text': """## 🌳 결정트리와 랜덤포레스트

### 결정트리 (Decision Tree)
데이터를 특정 기준으로 분할하여 트리 구조로 분류/예측하는 알고리즘입니다.

```python
from sklearn.tree import DecisionTreeClassifier

# 모델 생성 및 학습
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# 예측
predictions = dt.predict(X_test)
```

**장점**: 해석이 쉬움, 전처리 적음
**단점**: 과적합 위험

---

### 랜덤포레스트 (Random Forest)
여러 결정트리를 앙상블하여 성능을 향상시킨 알고리즘입니다.

```python
from sklearn.ensemble import RandomForestClassifier

# 100개의 트리로 앙상블
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 특성 중요도 확인
importance = rf.feature_importances_
```

**핵심 하이퍼파라미터**:
- `n_estimators`: 트리 개수
- `max_depth`: 최대 깊이
- `min_samples_split`: 분할 최소 샘플 수""",
        'sources': [
            {'type': 'IPYNB', 'title': '09_결정트리와 랜덤포레스트.ipynb', 'content': '트리 기반 모델'},
            {'type': 'IPYNB', 'title': '10_앙상블_부스팅.ipynb', 'content': '앙상블 기법'}
        ]
    },
    '딥러닝': {
        'text': """## 🧠 딥러닝 기초

딥러닝은 인공신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 기술입니다.

### 신경망의 기본 구조

```
입력층 → 은닉층(들) → 출력층
  ↓         ↓          ↓
특성값    가중치 연산    예측값
```

### 핵심 개념

1. **뉴런과 활성화 함수**
```python
# 활성화 함수 예시
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
```

2. **경사하강법 (Gradient Descent)**
- 손실 함수를 최소화하는 방향으로 가중치 업데이트
- 학습률(learning rate)이 중요

3. **역전파 (Backpropagation)**
- 출력층에서 입력층 방향으로 오차 전파
- 체인룰(Chain Rule)을 이용한 미분

### 간단한 신경망 예시
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 은닉층 구조
    activation='relu',
    max_iter=500
)
mlp.fit(X_train, y_train)
```""",
        'sources': [
            {'type': 'IPYNB', 'title': '13_선형모델_로지스틱회귀.ipynb', 'content': 'DEEPLEARNING 기초'},
            {'type': 'IPYNB', 'title': '11_최적화-경사하강법.ipynb', 'content': '경사하강법 원리'}
        ]
    }
}


def learning_agent(message, context=None):
    """
    학습용 에이전트
    
    [백엔드 연결 방법]
    USE_REAL_BACKEND = True로 설정 후,
    BACKEND_URL의 /chat 엔드포인트로 요청을 보내도록 수정하세요.
    """
    
    # 실제 백엔드 사용 시
    if USE_REAL_BACKEND:
        try:
            import requests
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"message": message, "context": context},
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {
                'text': f"⚠️ 백엔드 연결 오류: {str(e)}\n\n백엔드 서버가 실행 중인지 확인해주세요.",
                'sources': [],
                'steps': [{'step': 1, 'title': '오류 발생', 'desc': '백엔드 연결 실패'}]
            }
    
    # 샘플 응답 모드 (백엔드 연결 전)
    message_lower = message.lower()
    
    for keyword, response in SAMPLE_RESPONSES.items():
        if keyword in message_lower or keyword in message:
            return {
                'text': response['text'],
                'sources': response['sources'],
                'steps': [
                    {'step': 1, 'title': '질문 분석', 'desc': f'"{message}" 질문 파악'},
                    {'step': 2, 'title': '자료 검색', 'desc': '관련 ipynb 파일 탐색'},
                    {'step': 3, 'title': '답변 생성', 'desc': '학습 자료 기반 설명 작성'}
                ]
            }
    
    # 기본 응답
    return {
        'text': f"""## 📚 학습 도우미

**"{message}"**에 대해 알려드릴게요!

현재 다음 주제들에 대한 학습 자료가 준비되어 있습니다:

| 주제 | 관련 파일 |
|:---|:---|
| 🤖 머신러닝 개요 | 01_머신러닝개요.ipynb |
| 📊 데이터 전처리 | 04_데이터_전처리.ipynb |
| 🌳 결정트리/랜덤포레스트 | 09_결정트리와 랜덤포레스트.ipynb |
| 🧠 딥러닝 기초 | 13_선형모델_로지스틱회귀.ipynb |
| 📈 평가지표 | 05_평가지표.ipynb |
| 🔧 SVM | 07_지도학습_SVM.ipynb |

위 주제 중 하나를 선택해서 질문해주세요! 😊""",
        'sources': [
            {'type': 'INFO', 'title': '학습 자료 안내', 'content': '14개 ipynb 파일 기반'}
        ],
        'steps': [
            {'step': 1, 'title': '질문 분석', 'desc': f'"{message}" 질문 파악'},
            {'step': 2, 'title': '자료 검색', 'desc': '관련 학습 자료 탐색'},
            {'step': 3, 'title': '답변 생성', 'desc': '가이드 안내'}
        ]
    }


def get_agent_response(mode, message, context=None):
    """모드에 따라 적절한 에이전트 호출"""
    return learning_agent(message, context)


# ============================================
# 라우트 (Routes) - URL 엔드포인트 정의
# ============================================

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html', modes=MODES)


@app.route('/chat', methods=['POST'])
def chat():
    """채팅 API - POST /chat"""
    data = request.get_json()
    message = data.get('message', '')
    mode = data.get('mode', 'learning')
    
    # 에이전트에서 응답 생성
    response = get_agent_response(mode, message)
    
    return jsonify({
        'answer': response['text'],
        'sources': response['sources'],
        'steps': response['steps']
    })


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """스트리밍 채팅 API - POST /chat/stream"""
    data = request.get_json()
    message = data.get('message', '')
    mode = data.get('mode', 'learning')
    
    # 에이전트 응답 생성
    response = get_agent_response(mode, message)
    
    def generate():
        # 1단계: 진행 단계 정보 전송
        for step in response['steps']:
            yield f"data: {json.dumps({'type': 'step', 'data': step})}\n\n"
            time.sleep(0.5)
        
        # 2단계: 답변을 글자 하나씩 전송
        for char in response['text']:
            yield f"data: {json.dumps({'type': 'char', 'data': char})}\n\n"
            time.sleep(0.02)
        
        # 3단계: 참고 자료 전송
        yield f"data: {json.dumps({'type': 'sources', 'data': response['sources']})}\n\n"
        
        # 4단계: 완료 신호
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/reset', methods=['POST'])
def reset_all():
    """전체 세션 초기화"""
    session.clear()
    return jsonify({'success': True, 'message': 'Session reset'})


@app.route('/modes')
def get_modes():
    """사용 가능한 모드 목록"""
    return jsonify(MODES)


# ============================================
# 앱 실행
# ============================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)
