
from datetime import date
from langchain_core.prompts import ChatPromptTemplate
from src.agent.prompts import PROMPTS
from src.schema.state import AgentState
from langchain_openai import ChatOpenAI

class AnalystGeneratorNode:
    def __init__(self, query, tools, model_name):
        prompt = ChatPromptTemplate(
            [
                ("system", PROMPTS["ANALYSIS_SYSTEM_PROMPT"]),
                ("human", query)
            ],
            partial_variables={"today": date.today().strftime("%Y년 %m월 %d일")}
        )
        self.chain = prompt | ChatOpenAI(model=model_name).bind_tools(tools=tools)

    def __call__(self, state: AgentState):  # Node에서 호출할 call메소드.
        # state['messages']: [AIMessage, HumanMessage, AIMessage, ....] 마지막 Message객체를 조회
        response = self.chain.invoke({"query": state['analyst_results']})
        return response
