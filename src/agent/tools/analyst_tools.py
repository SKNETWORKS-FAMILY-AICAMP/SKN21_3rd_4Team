from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List

# 답변 구조 정의 (Schema)
class AnalystOutputSchema(BaseModel):
    summary: str = Field(description="[1] 핵심 개념 요약 (3~5줄)")
    code_explanation: str = Field(description="[2] 수업 코드 기준 설명")
    practice_tips: str = Field(description="[3] 실습 관점 팁")
    one_liner: str = Field(description="[4] 한 줄 정리")
    confidence_score: int = Field(description="답변 확신도 (0-100)")
    references: List[str] = Field(description="참고한 파일명 리스트")
    suggested_questions: List[str] = Field(description="[5] 사용자가 이어서 질문할 만한 연관 질문 3개")


# args_schema를 지정하여 LLM에게 입력 형식을 알려줍니다.
@tool(args_schema=AnalystOutputSchema)
def submit_analysis(summary: str, code_explanation: str, practice_tips: str, one_liner: str, confidence_score: int, references: List[str], suggested_questions: List[str]):
    """
    최종 분석 결과를 제출할 때 사용하는 도구입니다.
    LLM은 답변을 텍스트로 바로 뱉는 대신, 이 도구를 호출하여 구조화된 답변을 전달해야 합니다.
    """
    # 실제로는 여기서 받은 데이터를 가공하거나 그대로 반환하면 됩니다.
    return {
        "summary": summary,
        "code_explanation": code_explanation,
        "practice_tips": practice_tips,
        "one_liner": one_liner,
        "confidence_score": confidence_score,
        "references": references,
        "suggested_questions": suggested_questions
    }