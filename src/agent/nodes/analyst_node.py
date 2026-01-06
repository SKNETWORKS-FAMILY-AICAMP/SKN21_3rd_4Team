# src/agent/nodes/analyst_node.py

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langgraph.graph import END
from typing import Literal

from src.agent.prompts.analyst_prompt import analyst_prompt

from langgraph.types import Command
from langgraph.graph import END
from langchain_core.messages import HumanMessage

from src.agent.prompts.analyst_prompt import analyst_prompt

def build_analyst_node(llm):

    def analyst_node(state):
        question = state["messages"][-1].content
        context = state.get("context", "")

        prompt_value = analyst_prompt.invoke({
            "question": question,
            "context": context
        })

        response = llm.invoke(prompt_value)

        return Command(
            update={
                "messages": state["messages"] + [
                    AIMessage(
                        content=response.content,
                        name="analyst"
                    )
                ]
            },
            goto=END
        )

    return analyst_node

