# Query Enhancement Module (Advanced RAG)
# 쿼리 확장, 동의어 매핑, 약어 확장 등을 제공하는 모듈

from typing import Dict, List, Optional


# ============================================================
# 1. 키워드 확장 매핑 (Query Expansion)
# ============================================================

KEYWORD_EXPANSION: Dict[str, str] = {
    # 머신러닝 기초
    "머신러닝": "기계학습 ML machine learning 인공지능 AI 지도학습 비지도학습 강화학습",
    "딥러닝": "deep learning DL 신경망 neural network 인공지능 MLP CNN RNN",
    
    # 데이터 전처리
    "전처리": "preprocessing 결측치 이상치 스케일링 정규화 인코딩",
    "결측치": "missing value NaN null 빈값 처리",
    "스케일링": "scaling 정규화 normalization 표준화 standardization MinMaxScaler",
    
    # 평가 지표
    "평가지표": "evaluation metric 정확도 accuracy 정밀도 precision 재현율 recall F1",
    "정확도": "accuracy 성능 평가",
    "과적합": "overfitting 과대적합 일반화 regularization 드롭아웃 dropout",
    
    # 머신러닝 알고리즘
    "결정트리": "decision tree 트리 분기 지니 gini 엔트로피 entropy",
    "랜덤포레스트": "random forest 앙상블 ensemble 배깅 bagging",
    "SVM": "서포트 벡터 머신 support vector machine 커널 kernel 마진",
    "KNN": "k-최근접 이웃 k-nearest neighbors 거리 distance",
    "로지스틱회귀": "logistic regression 이진분류 시그모이드 sigmoid",
    "선형회귀": "linear regression 회귀 regression 예측",
    
    # 앙상블
    "앙상블": "ensemble 부스팅 boosting 배깅 bagging XGBoost LightGBM",
    "부스팅": "boosting gradient boosting XGBoost LightGBM AdaBoost",
    "XGBoost": "xgboost 부스팅 앙상블 gradient boosting",
    
    # 최적화
    "경사하강법": "gradient descent SGD 최적화 optimization 학습률 learning rate Adam",
    "그리드서치": "grid search 하이퍼파라미터 hyperparameter 튜닝 tuning",
    
    # 군집
    "군집": "clustering 클러스터링 K-means DBSCAN 비지도학습",
    
    # 검증
    "교차검증": "cross validation cv k-fold 폴드 검증",
    "train test": "훈련 테스트 분할 split validation",
    
    # ============================================================
    # 딥러닝 관련 키워드 (추가)
    # ============================================================
    
    # 신경망 기초
    "신경망": "neural network 뉴런 neuron 퍼셉트론 perceptron 활성화함수",
    "MLP": "다층 퍼셉트론 Multi-Layer Perceptron 완전연결층 fully connected",
    "활성화함수": "activation function ReLU sigmoid tanh softmax",
    
    # CNN
    "CNN": "합성곱 신경망 Convolutional Neural Network 이미지 분류 conv2d",
    "합성곱": "convolution CNN 필터 filter 커널 kernel 풀링 pooling",
    "풀링": "pooling max pooling average pooling 다운샘플링",
    
    # RNN
    "RNN": "순환 신경망 Recurrent Neural Network 시퀀스 sequence LSTM GRU",
    "LSTM": "Long Short-Term Memory 장단기 메모리 순환 신경망 시계열",
    
    # PyTorch
    "파이토치": "pytorch torch 딥러닝 프레임워크 텐서",
    "텐서": "tensor pytorch 다차원 배열 numpy",
    "DataLoader": "데이터로더 Dataset 배치 batch 학습 데이터",
    
    # 딥러닝 학습
    "역전파": "backpropagation 오차역전파 gradient 체인룰",
    "손실함수": "loss function 비용함수 cost function MSE CrossEntropy",
    "옵티마이저": "optimizer 최적화 SGD Adam RMSprop",
    "에포크": "epoch 학습 반복 iteration batch",
    
    # 성능 개선
    "배치정규화": "batch normalization BN 정규화 학습 안정화",
    "드롭아웃": "dropout 과적합 방지 regularization",
    "조기종료": "early stopping 과적합 방지 검증 손실",
    
    # Universal Approximation Theorem
    "보편근사정리": "Universal Approximation Theorem 신경망 근사 함수 표현",
    
    # ============================================================
    # 강의 자료 기반 추가 키워드
    # ============================================================
    
    # 데이터셋
    "Iris": "아이리스 붓꽃 데이터셋 분류 sklearn 첫번째 머신러닝",
    "데이터셋": "dataset 학습 데이터 train test 분할",
    
    # 모델 관리
    "모델저장": "model save load pickle joblib 저장 불러오기",
    "파이프라인": "pipeline sklearn 전처리 모델 연결 workflow",
    
    # 특성/피처
    "특성중요도": "feature importance 변수 중요도 랜덤포레스트 트리",
    "피처엔지니어링": "feature engineering 특성 생성 변환",
    
    # 문제 유형
    "분류": "classification 이진분류 다중분류 binary multi-class",
    "회귀": "regression 연속값 예측 선형 비선형",
    
    # 평가 관련
    "혼동행렬": "confusion matrix 오차 행렬 TP TN FP FN",
    "F1점수": "F1 score 정밀도 재현율 조화평균",
    "ROC커브": "ROC curve AUC 분류 성능 곡선",
    
    # 일반화
    "일반화": "generalization 과적합 과소적합 편향 분산",
    "편향분산": "bias variance tradeoff 과적합 과소적합",
}


# ============================================================
# 2. 동의어 매핑 (양방향)
# ============================================================

SYNONYMS: Dict[str, List[str]] = {
    "머신러닝": ["기계학습", "ML", "machine learning"],
    "딥러닝": ["deep learning", "DL", "심층학습"],
    "인공지능": ["AI", "artificial intelligence"],
    "과적합": ["overfitting", "과대적합"],
    "정규화": ["regularization", "normalization"],
    "정확도": ["accuracy", "acc"],
    "손실함수": ["loss function", "cost function", "비용함수"],
    "하이퍼파라미터": ["hyperparameter", "하이파", "hp"],
    "전처리": ["preprocessing", "데이터 전처리"],
    "결측치": ["missing value", "NaN", "null", "빈값"],
    "이상치": ["outlier", "특이값"],
    "피처": ["feature", "특성", "변수"],
    "레이블": ["label", "타겟", "target", "정답"],
    
    # 딥러닝 관련 (추가)
    "신경망": ["neural network", "NN"],
    "합성곱": ["convolution", "conv"],
    "텐서": ["tensor", "배열"],
    "파이토치": ["pytorch", "torch"],
    "에포크": ["epoch", "반복"],
    "배치": ["batch", "미니배치"],
    "활성화함수": ["activation function", "활성 함수"],
}


# ============================================================
# 3. 약어 확장
# ============================================================

ABBREVIATIONS: Dict[str, str] = {
    "ML": "머신러닝 Machine Learning",
    "DL": "딥러닝 Deep Learning",
    "AI": "인공지능 Artificial Intelligence",
    "SVM": "서포트 벡터 머신 Support Vector Machine",
    "KNN": "K-최근접 이웃 K-Nearest Neighbors",
    "CNN": "합성곱 신경망 Convolutional Neural Network",
    "RNN": "순환 신경망 Recurrent Neural Network",
    "NLP": "자연어 처리 Natural Language Processing",
    "CV": "교차 검증 Cross Validation",
    "MSE": "평균 제곱 오차 Mean Squared Error",
    "MAE": "평균 절대 오차 Mean Absolute Error",
    "RMSE": "평균 제곱근 오차 Root Mean Squared Error",
    "ROC": "수신자 조작 특성 Receiver Operating Characteristic",
    "AUC": "곡선 하 면적 Area Under Curve",
    
    # 딥러닝 관련 (추가)
    "MLP": "다층 퍼셉트론 Multi-Layer Perceptron",
    "LSTM": "장단기 메모리 Long Short-Term Memory",
    "GRU": "게이트 순환 유닛 Gated Recurrent Unit",
    "SGD": "확률적 경사하강법 Stochastic Gradient Descent",
    "BN": "배치 정규화 Batch Normalization",
    "GPU": "그래픽 처리 장치 Graphics Processing Unit",
}


# ============================================================
# 4. 쿼리 확장 함수
# ============================================================

def expand_query(query: str) -> str:
    """
    쿼리 확장: 관련 키워드를 추가하여 검색 정확도 향상
    
    Args:
        query: 원본 사용자 질문
        
    Returns:
        확장된 쿼리 (원본 + 관련 키워드)
        
    Example:
        >>> expand_query("머신러닝이 뭐야?")
        "머신러닝이 뭐야? 기계학습 ML machine learning 인공지능 AI 지도학습 비지도학습 강화학습"
    """
    expanded_parts = [query]
    
    # 키워드 확장 적용
    for keyword, expansion in KEYWORD_EXPANSION.items():
        if keyword in query:
            expanded_parts.append(expansion)
            break  # 첫 번째 매칭만 적용 (과도한 확장 방지)
    
    return " ".join(expanded_parts)


def expand_abbreviation(query: str) -> str:
    """
    약어 확장: 약어를 풀 네임으로 변환
    
    Args:
        query: 원본 질문
        
    Returns:
        약어가 확장된 질문
    """
    words = query.split()
    expanded_words = []
    
    for word in words:
        upper_word = word.upper()
        if upper_word in ABBREVIATIONS:
            expanded_words.append(f"{word}({ABBREVIATIONS[upper_word]})")
        else:
            expanded_words.append(word)
    
    return " ".join(expanded_words)


def get_synonyms(term: str) -> List[str]:
    """
    동의어 가져오기
    
    Args:
        term: 검색할 용어
        
    Returns:
        동의어 리스트 (없으면 빈 리스트)
    """
    return SYNONYMS.get(term, [])


def enhance_query(query: str, expand: bool = True, abbreviate: bool = False) -> str:
    """
    통합 쿼리 향상 함수
    
    Args:
        query: 원본 질문
        expand: 키워드 확장 적용 여부
        abbreviate: 약어 확장 적용 여부
        
    Returns:
        향상된 쿼리
    """
    result = query
    
    if abbreviate:
        result = expand_abbreviation(result)
    
    if expand:
        result = expand_query(result)
    
    return result
