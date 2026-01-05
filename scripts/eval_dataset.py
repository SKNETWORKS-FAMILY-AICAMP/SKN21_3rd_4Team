"""
평가 데이터셋 (Ground Truth)

각 질문에 대해 기대되는 정답 파일을 지정합니다.
정확도 평가는 검색된 문서에 해당 파일이 포함되어 있는지로 판단합니다.
"""

EVALUATION_DATASET = [
    # 머신러닝 기초 개념
    {
        "question": "과적합이란 무엇인가?",
        "expected_files": ["06_과적합_일반화_그리드서치_파이프라인.ipynb"],
        "topic": "머신러닝기초",
    },
    {
        "question": "교차검증이란 무엇인가?",
        "expected_files": ["03_데이터셋 나누기와 모델검증.ipynb"],
        "topic": "모델검증",
    },
    {
        "question": "일반화란 무엇인가?",
        "expected_files": ["06_과적합_일반화_그리드서치_파이프라인.ipynb"],
        "topic": "머신러닝기초",
    },
    {
        "question": "그리드서치는 무엇인가?",
        "expected_files": ["06_과적합_일반화_그리드서치_파이프라인.ipynb"],
        "topic": "하이퍼파라미터",
    },
    
    # 알고리즘 관련
    {
        "question": "SVM이란 무엇인가?",
        "expected_files": ["07_지도학습_SVM.ipynb"],
        "topic": "지도학습",
    },
    {
        "question": "랜덤포레스트 설명해줘",
        "expected_files": ["09_결정트리와 랜덤포레스트.ipynb"],
        "topic": "앙상블",
    },
    {
        "question": "결정트리란 무엇인가?",
        "expected_files": ["09_결정트리와 랜덤포레스트.ipynb", "02_첫번째 머신러닝 분석 - Iris_분석.ipynb"],
        "topic": "지도학습",
    },
    {
        "question": "KNN 알고리즘 설명해줘",
        "expected_files": ["08_지도학습_최근접이웃.ipynb"],
        "topic": "지도학습",
    },
    {
        "question": "부스팅이란 무엇인가?",
        "expected_files": ["10_앙상블_부스팅.ipynb"],
        "topic": "앙상블",
    },
    
    # 데이터 전처리
    {
        "question": "결측치 처리 방법",
        "expected_files": ["04_데이터_전처리.ipynb"],
        "topic": "전처리",
    },
    {
        "question": "이상치 처리 방법",
        "expected_files": ["04_데이터_전처리.ipynb"],
        "topic": "전처리",
    },
    {
        "question": "Feature Scaling이란?",
        "expected_files": ["04_데이터_전처리.ipynb"],
        "topic": "전처리",
    },
    
    # 평가지표
    {
        "question": "정확도란 무엇인가?",
        "expected_files": ["05_평가지표.ipynb"],
        "topic": "평가",
    },
    {
        "question": "정밀도와 재현율의 차이",
        "expected_files": ["05_평가지표.ipynb"],
        "topic": "평가",
    },
    {
        "question": "F1 Score란?",
        "expected_files": ["05_평가지표.ipynb"],
        "topic": "평가",
    },
    
    # 회귀/분류
    {
        "question": "선형회귀란 무엇인가?",
        "expected_files": ["12_선형모델_선형회귀.ipynb"],
        "topic": "회귀",
    },
    {
        "question": "로지스틱 회귀 설명해줘",
        "expected_files": ["13_선형모델_로지스틱회귀.ipynb"],
        "topic": "분류",
    },
    
    # 비지도학습
    {
        "question": "군집화란 무엇인가?",
        "expected_files": ["14 군집_Clustering.ipynb"],
        "topic": "비지도학습",
    },
    
    # 최적화
    {
        "question": "경사하강법이란?",
        "expected_files": ["11_최적화-경사하강법.ipynb"],
        "topic": "최적화",
    },
    
    # 기초 개념
    {
        "question": "머신러닝이란 무엇인가?",
        "expected_files": ["01_머신러닝개요.ipynb"],
        "topic": "개요",
    },
    {
        "question": "지도학습과 비지도학습의 차이",
        "expected_files": ["01_머신러닝개요.ipynb"],
        "topic": "개요",
    },
    
    # ==============================================
    # Python Documentation 관련 질문
    # ==============================================
    {
        "question": "파이썬 리스트 메서드 종류",
        "expected_files": ["library/stdtypes.txt"],
        "topic": "Python기초",
    },
    {
        "question": "파이썬 딕셔너리 사용법",
        "expected_files": ["library/stdtypes.txt"],
        "topic": "Python기초",
    },
    {
        "question": "파이썬 for문 사용법",
        "expected_files": ["reference/compound_stmts.txt"],
        "topic": "Python제어문",
    },
    {
        "question": "파이썬 클래스 정의 방법",
        "expected_files": ["reference/compound_stmts.txt"],
        "topic": "Python객체지향",
    },
    {
        "question": "파이썬 예외 처리 try except",
        "expected_files": ["reference/compound_stmts.txt"],
        "topic": "Python예외처리",
    },
    {
        "question": "파이썬 lambda 함수",
        "expected_files": ["reference/expressions.txt"],
        "topic": "Python함수",
    },
    {
        "question": "파이썬 list comprehension",
        "expected_files": ["reference/expressions.txt"],
        "topic": "Python기초",
    },
    {
        "question": "파이썬 import 사용법",
        "expected_files": ["reference/simple_stmts.txt"],
        "topic": "Python모듈",
    },
    {
        "question": "파이썬 문자열 메서드",
        "expected_files": ["library/stdtypes.txt"],
        "topic": "Python기초",
    },
    {
        "question": "파이썬 range 함수",
        "expected_files": ["library/stdtypes.txt"],
        "topic": "Python기초",
    },
]
