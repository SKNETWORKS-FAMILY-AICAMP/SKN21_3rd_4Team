from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch

from src.utils.config import ConfigLLM
from src.agent.prompts import PROMPTS
from src.agent.tools.analyst_tools import submit_analysis
from src.schema.state import AgentState

def check_relevance(state: AgentState):
    search_results = state['search_results']
    
    # 검색 결과가 없으면 고정 메시지
    if not search_results:
        return "no_data_node"
    
    # 평균 점수 계산
    scores = [r['score'] for r in search_results]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print(f"📊 [check_relevance] 평균 유사도: {avg_score:.3f} (문서 {len(scores)}개)")
    
    # 3단계 분기 (평균 점수 기준)
    if avg_score <= 0.3:
        # 평균 유사도 너무 낮음 → 고정 메시지
        print("   → no_data_node (평균 ≤ 0.3)")
        return "no_data_node"
    elif avg_score <= 0.5:
        # 중간 평균 유사도 → Tavily 웹 검색 추가
        print("   → web_search_node (0.3 < 평균 ≤ 0.5)")
        return "web_search_node"
    else:
        # 높은 평균 유사도 → Qdrant만 사용
        print("   → analyst_node (평균 > 0.5)")
        return "analyst_node"


def analyst_node(state: AgentState):
    # 1. Prompt 정의 (System Message + Human Message)    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(PROMPTS["ANALYSIS_SYSTEM_PROMPT"]),
        HumanMessagePromptTemplate.from_template("{query}")
    ])

    # 2. LLM 설정
    llm = ChatOpenAI(
        model=ConfigLLM.OPENAI_MODEL,
        temperature=0
    ).bind_tools([submit_analysis], tool_choice="submit_analysis")
    
    # 4. Chain 연결 (Prompt -> LLM)
    chain = prompt | llm

    # 5. 실행 (state에 있는 'query', 'context' 등이 prompt의 변수로 주입됨)
    # invoke 시 state(dict) 전달
    response = chain.invoke(state)

    tool_calls = response.tool_calls
    print(">>>> analyst_node : tool_calls", tool_calls)

    if tool_calls:
        response_text = str(tool_calls[0]['args'])
        # tool_calls에서 suggested_questions 추출
        questions = tool_calls[0]['args'].get('suggested_questions', [])
        print(f"💡 [analyst_node] 연관 질문: {questions}")
    else:
        response_text = "도구를 호출하지 않았습니다."
        questions = []

    # 6. 결과 반환
    return {
        "analyst_results": [
            HumanMessage(content=response_text, name="analyst")
        ],
        "messages": [AIMessage(content=response_text)],  # 대화 기록 저장
        "suggested_questions": questions
    }


def web_search_node(state: AgentState):
    """
    내부 문서 점수가 낮을 때 실행되는 외부 웹 검색 노드
    Tavily API를 사용하여 최신 정보를 검색하고 Context에 추가합니다.
    """
    query = state['query']
    
    tavily_search = TavilySearch(
                                    max_results=3,
                                )
    
    try:
        search_results = tavily_search.invoke(state)
    except Exception as e:
        # 에러 발생 시 빈 리스트 처리 (흐름 끊기지 않게)
        print(f"Web Search Error: {e}")
        search_results = []
    
    # Tavily 결과가 문자열인 경우 처리
    if isinstance(search_results, str):
        # 문자열이면 그대로 context에 추가
        web_context_str = f"[External Web] {search_results}"
    else:
        # 리스트인 경우 기존 로직
        web_context_parts = []
        for i, res in enumerate(search_results, 1):
            if isinstance(res, dict):
                content = res.get('content', '')
                url = res.get('url', '')
            else:
                content = str(res)
                url = ''
            part = f"[External Web {i}] 출처: {url}\n{content}"
            web_context_parts.append(part)
        web_context_str = "\n\n".join(web_context_parts)
    
    # 기존 build_context에서 만들어진 state['context'] 뒤에 추가
    current_context = state['context']
    web_context_str = "=== 외부 검색 결과 (Low Confidence Fallback, Weight: 0.3) ===\n" + web_context_str
    if current_context:
        new_context = current_context + "\n\n" + web_context_str
    else:
        new_context = web_context_str
        
    return {
        "context": new_context,
    }

def no_data_node(state: AgentState):
    """
    유사도가 너무 낮을 때 (0.3 이하) 실행되는 노드
    GPT를 호출하지 않고 고정 메시지 + 랜덤 추천 질문 반환
    """
    from langchain_core.messages import HumanMessage
    import random
    
    fixed_message = "❌ 데이터에는 없는 자료입니다.\n\n학습 자료와 관련된 질문을 해주세요!"
    
    # 추천 질문 풀 (약 30개)
    question_pool = [
        # Python
        "파이썬 리스트와 튜플의 차이점은?",
        "파이썬에서 딕셔너리 정렬하는 방법",
        "파이썬 lambda 함수 사용법 알려줘",
        "파이썬 예외처리 try-except 사용법",
        "파이썬 __init__ 메서드의 역할은?",
        "파이썬 데코레이터(Decorator)가 뭐야?",
        "파이썬 제너레이터(Generator) 설명해줘",
        "파이썬 가상환경은 왜 사용해야 해?",
        "파이썬 map과 filter 함수 사용법",
        "파이썬 클래스 상속하는 방법",
        
        # Machine Learning
        "머신러닝과 딥러닝의 차이점이 뭐야?",
        "지도학습과 비지도학습의 차이는?",
        "과적합(Overfitting)을 방지하는 방법은?",
        "정밀도(Precision)와 재현율(Recall) 설명해줘",
        "경사하강법(Gradient Descent)이란?",
        "랜덤 포레스트(Random Forest) 모델 설명해줘",
        "SVM(Support Vector Machine) 알고리즘 원리",
        "K-평균(K-Means) 클러스터링이란?",
        "교차 검증(Cross Validation)이 뭐야?",
        "앙상블(Ensemble) 기법에는 어떤 게 있어?",
        
        # Deep Learning
        "CNN(Convolutional Neural Network)이 뭐야?",
        "RNN(Recurrent Neural Network)의 특징은?",
        "활성화 함수(Activation Function) 종류 알려줘",
        "Relu 함수를 사용하는 이유는?",
        "배치 정규화(Batch Normalization)란?",
        "드롭아웃(Dropout)의 효과는?",
        "전이 학습(Transfer Learning)이 뭐야?",
        "역전파(Backpropagation) 알고리즘 설명해줘",
        "딥러닝에서 Epoch, Batch Size 의미",
        "Transformer 모델의 주요 특징은?"
    ]
    
    # 랜덤으로 3개 선택
    suggested_questions = random.sample(question_pool, 3)
    
    return {
        "context": "",
        "search_results": [],
        "suggested_questions": suggested_questions,
        "analyst_results": [
            HumanMessage(content=fixed_message, name="analyst")
        ],
        "messages": [AIMessage(content=fixed_message)]  # 대화 기록 저장
    }