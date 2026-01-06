from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from src.agent.prompts import PROMPTS
from src.schema.state import AgentState

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
    )
    
    # 4. Chain 연결 (Prompt -> LLM -> Parser)
    chain = prompt | llm | StrOutputParser()

    # 5. 실행 (state에 있는 'query', 'context' 등이 prompt의 변수로 주입됨)
    # invoke 시 state(dict) 전달
    response_text = chain.invoke(state, {"query": state["query"]})

    # 6. 결과 반환
    # state 정의상 analyst_results는 Annotated[list, add_messages] 타입으로 추정
    return {
        "analyst_results": [
            HumanMessage(content=response_text, name="analyst")
        ]
    }
