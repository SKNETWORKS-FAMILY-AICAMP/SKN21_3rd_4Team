from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from typing import Literal
from langgraph.graph import END

from agent.prompts.analyst_prompt import analyst_prompt

def build_analyst_node(llm):

    analyst_agent = create_react_agent(
        llm=llm,
        tools=[],   # 분석만 할땐 tool X 
        prompt=analyst_prompt
    )

    def analyst_node(state) -> Command[Literal[END]]:
        result = analyst_agent.invoke(state)

        # provider 호환을 위해 HumanMessage로 wrapping
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content,
            name="analyst"
        )

        return Command(
            update={"messages": result["messages"]},
            goto=END
        )

    return analyst_node
