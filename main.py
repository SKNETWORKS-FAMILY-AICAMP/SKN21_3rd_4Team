# 과거 메시지 관리 필요

def main():
    '''
    app.py 에서 호출해서 질문을 넘겨줘야함.
    graph.stream()이나 invoke()를 사용하여 workflow를 실행합니다.
    return events에대해 app.py로 넘겨줘야함.
    '''
    events = graph.stream(
        {
            "messages": [
                (
                    "user",
                    "",
                    ""
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    )
    
    for s in events:
        print(s)
        print("----")

    return events