from langchain_core.messages import HumanMessage
from langgraph.types import Command
from typing import Literal
from langgraph.graph import END

def search_node(state) -> Command[Literal["analyst"]]:
    # search 결과 (JSON)
    search_result = {
        "query": "pathlib.Path 사용법 알려줘",
        "retrieved_documents": [
            {
                "content": "pathlib.Path is an object-oriented filesystem path API.",
                "metadata": {"source": "python_doc"},
                "score": 0.56
            }
        ]
    }

    message = HumanMessage(
        content=f"""
[SEARCH RESULT]
{search_result}
""",
        name="search"
    )

    return Command(
        update={"messages": state["messages"] + [message]},
        goto="analyst"
    )
