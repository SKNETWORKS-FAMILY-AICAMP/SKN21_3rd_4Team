from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.agent.graph import build_graph

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini")

graph = build_graph(llm)

result = graph.invoke({
    "messages": [
        HumanMessage(content="pathlib.Path 사용법 알려줘")
    ]
})

# for msg in result["messages"]:
#     print(f"[{msg.name}] {msg.content}")

answer = result["messages"][-1].content
print(answer)