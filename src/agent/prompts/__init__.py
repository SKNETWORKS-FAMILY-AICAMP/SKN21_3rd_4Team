# import문
# >>>>  from src.agent.prompts import PROMPTS
# import 후에 PROMPTS 에서 각 프롬프트 꺼내는 방법
# >>>>  PROMPTS["ANALYSIS_SYSTEM_PROMPT"] : key값으로 불러서 사용

from src.agent.prompts.analyst_prompt import ANALYSIS_SYSTEM_PROMPT
from src.agent.prompts.search_prompt import SEARCH_ROUTER_PROMPT
from src.agent.prompts.translate_prompt import TRANSLATE_PROMPT

PROMPTS = {
    "ANALYSIS_SYSTEM_PROMPT": ANALYSIS_SYSTEM_PROMPT,
    "SEARCH_ROUTER_PROMPT": SEARCH_ROUTER_PROMPT,
    "TRANSLATE_PROMPT": TRANSLATE_PROMPT,
}