from src.schema.state import AgentState

class AnalystToolNode:
    '''
    AIMessage의 tool_calls를 이용해 tool을 호출학고 그 결과를 반환.
    '''
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name:tool  for tool in tools} # {툴이름: 툴객체}

    def __call__(self, state: AgentState):
        messages = state.get('analyst_results', []) # State에 messages가 없으면 빈 리스트 반환
        if messages:
            message = messages[-1]
        else:
            raise ValueError("State에 Message가 없습니다.")
        outputs = [] # Tool Calling 결과들을 담을 빈 리스트 -> [ToolMessage, ToolMessage, ..]
        for tool_call in message.tool_calls:
            # TOOL을 호출 - tool객체.invoke(tool_call)
            tool_message = self.tools_by_name[tool_call["name"]].invoke(tool_call)
            outputs.append(tool_message)
        
        return {"analyst_results": outputs} # State의 messages에 추가
