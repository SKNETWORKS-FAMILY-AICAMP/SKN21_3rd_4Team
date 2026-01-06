# 과거 메시지 관리 필요

from src.agent.workflow import build_graph
from langgraph.checkpoint.memory import MemorySaver # 메모리 이용
from src.schema.state import AgentState
from pprint import pprint

def main(query: str):
    '''
    app.py 에서 호출해서 질문을 넘겨줘야함.
    graph.stream()이나 invoke()를 사용하여 workflow를 실행합니다.
    return events에대해 app.py로 넘겨줘야함.
    '''

    checkpointer = MemorySaver()

    graph = build_graph()
    graph = graph.compile(checkpointer=checkpointer)

    agent_state = AgentState(
        query=query,
        messages=[("human", query)],
    )

    config = {"configurable": {"thread_id": "1"}}  # 고유한 ID 지정
    response = graph.invoke(agent_state, config=config)
    pprint(response)
    
    return response

if __name__ == "__main__":
    main("AI의 세가지 특징을 정의해줘.")