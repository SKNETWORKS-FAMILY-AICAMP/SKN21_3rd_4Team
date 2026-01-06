# LLM 답변 생성 로직
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.schema.state import AgentState
from src.agent.prompts import PROMPTS
from src.agent.nodes.analyst_tools import AnalystToolNode
from src.agent.nodes.analyst_generator import AnalystGeneratorNode


def analyst_node(state: AgentState):
    result = generetor_node(state)
    print(result)


def generetor_node(state: AgentState):
    result = AnalystGeneratorNode(state['query'], [], "gpt-4o-mini")
    print(type(result))
    return result


def grounding_node(state: AgentState):
    tool_node = AnalystToolNode([])
    result = tool_node.invoke(state)

    return result


if __name__ == "__main__":
    state = AgentState()
    state['context'] = """
파이썬 인터프리터에는 항상 사용할 수 있는 많은 함수와 형이 내장되어 있\n습니다. 여기에서 알파벳 순으로 나열합니다.
파이썬 프로그램은 코드 블록으로 만들어집니다. *블록 (block)* 은 한 단\n위로 실행되는 한 조각의 파이썬 프로그램 텍스트입니다. 다음과 같은 것들\n이 블록입니다: 모듈, 함수 바디, 클래스 정의. 대화형으로 입력되는 각 명\n령은 블록입니다. 스크립트 파일(표준 입력을 통해 인터프리터로 제공되는\n파일이나 인터 프리터에 명령행 인자로 지정된 파일)은 코드 블록입니다. 스\n크립트 명령(\"-c\" 옵션으로 인터프리터 명령행에 지정된 명령)은 코드 블록\n입니다. \"-m\" 인자를 사용하여 명령 줄에서 최상위 수준 스크립트로 (모듈\n\"__main__\"으로) 실행되는 모듈도 코드 블록입니다. 내장함수 \"eval()\" 과\n\"exec()\" 로 전달되는 문자열 인자도 코드 블록입니다.\n\n코드 블록은 *실행 프레임 (execution frame)* 에서 실행됩니다. 프레임은\n몇몇 관리를 위한 정보(디버깅에 사용됩니다)를 포함하고, 코드 블록의  실\n행이 끝난 후에 어디서 어떻게 실행을 계속할 것인지를 결정합니다.
파이썬 언어 레퍼런스 는 파이썬 언어의 정확한 문법과 의미를 설명하고 있\n지만, 이 라이브러리 레퍼런스 설명서는 파이썬과 함께 배포되는 표준 라 이\n브러리를 설명합니다. 또한, 파이썬 배포판에 일반적으로 포함되어있는 선\n택적 구성 요소 중 일부를 설명합니다.\n\n파이썬의 표준 라이브러리는 매우 광범위하며, 아래 나열된 긴 목차에 표시\n된 대로 다양한 기능을 제공합니다. 라이브러리에는 일상적인 프로그래밍에\n서 발생하는 많은 문제에 대한 표준적인 해결책을 제공하는 파 이썬으로 작\n성된 모듈뿐만 아니라, 파일 I/O와 같은 시스템 기능에 액세스하는 (C로 작\n성된) 내장 모듈들이 포함됩니다 (이 모듈들이 없다면 파이썬 프로그래머가\n액세스할 방법은 없습니다). 이 모듈 중 일부는 플랫폼 관련 사항을 플랫폼\n중립적인 API들로 추상화시킴으로써, 파이썬 프로그램의 이식성을 권장하고\n개선하도록 명시적으로 설계되었습니다.\n\n윈도우 플랫폼용 파이썬 설치 프로그램은 일반적으로 전체 표준 라이브러리\n를 포함하며 종종 많은 추가 구성 요소도 포함합니다. 유닉스와 같 은 운영\n체제의 경우, 파이썬은 일반적으로 패키지 모음으로 제공되기 때문에, 운영\n체제와 함께 제공되는 패키지 도구를 사용하여 선택적 구성 요소의 일부 또\n는 전 부를 구해야 할 수 있습니다.\n\n표준 라이브러리 외에도, 수십만 가지 컴포넌트(개별 프로그램과 모듈부터\n패키지 및 전체 응용 프로그램 개발 프레임워크까지)로 구성 된 활발한 모음\n이 있는데, 파이썬 패키지 색인 에서 얻을 수 있습니다.\n\n* 소개\n\n  * 가용성에 대한 참고 사항\n\n* 내장 함수\n\n* 내장 상수\n\n  * \"site\" 모 듈에 의해 추가된 상수들\n\n* 내장형\n\n  * 논리값 검사\n\n  * 논리 연산 --- \"and\", \"or\", \"not\"\n\n  * 비교\n\n  * 숫자 형 --- \"int\", \"float\", \"complex\"\n\n  * Boolean Type - \"bool\"\n\n  * 이터레이터 형
"""
    analyst_node(state)