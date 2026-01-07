from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

from src.agent.prompts import PROMPTS
from src.agent.tools.analyst_tools import submit_analysis
from src.schema.state import AgentState

def check_relevance(state: AgentState):
    search_results = state['search_results']

    # documents가 너무 짧으면 'web_search_node'로 이동
    if len(search_results) < 3:
        return "web_search_node"

    # 상위 3개 문서의 평균 점수 또는 최고 점수 확인
    scores = [r['score'] for r in search_results]
    max_score = max(scores) if scores else 0
    
    # 임계값 0.4 미만이면 'web_search_node'로, 아니면 'analyst_node'로 이동
    if max_score < 0.4:
        return "web_search_node"

    return "analyst_node"


def analyst_node(state: AgentState):
    # 1. Prompt 정의 (System Message + Human Message)    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(PROMPTS["ANALYSIS_SYSTEM_PROMPT"]),
        HumanMessagePromptTemplate.from_template("{query}")
    ])

    # 2. LLM 설정
    llm = ChatOpenAI(
        model="gpt-4o-mini",
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
    else:
        response_text = "도구를 호출하지 않았습니다."

    # 6. 결과 반환
    return {
        "analyst_results": [
            HumanMessage(content=response_text, name="analyst")
        ]
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
