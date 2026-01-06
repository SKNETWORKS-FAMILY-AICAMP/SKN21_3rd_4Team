import numpy as np

from src.schema.state import AgentState
from sentence_transformers import CrossEncoder

# 모델 로드 (최초 1회만 실행)
_RERANKER_MODEL = None

def _load_reranker_model():
    global _RERANKER_MODEL
    
    if _RERANKER_MODEL is None:
        _RERANKER_MODEL = CrossEncoder('BAAI/bge-reranker-v2-m3')

    return _RERANKER_MODEL


def rerank_node(state: AgentState):
    """
    Search Node에서 가져온 문서들을 질문과의 관련성 기준으로 재정렬한다.
    """
    query = state['query']
    results = state['search_results']

    model = _load_reranker_model()
    
    if not results:
        return {"search_results": []}

    # (Query, Document) 쌍 만들기
    # CrossEncoder는 [질문, 문서내용]을 한 쌍으로 입력받아 점수를 뱉습니다.
    passages = [r['content'] for r in results]
    predict_inputs = [[query, passage] for passage in passages]
    
    scores = model.predict(predict_inputs) # 점수 예측 (로짓값)
    scores = [(1 / (1 + np.exp(-score))) for score in scores] # Sigmoid 함수 적용하여 0~1 사이값으로 변환

    # 점수 업데이트 및 정렬
    for i, res in enumerate(results):
        res['score'] = float(scores[i])
        
    # 점수가 높은 순으로 내림차순 정렬
    reranked_results = sorted(results, key=lambda x: x['score'], reverse=True) 
    
    return {
        "search_results": reranked_results
    }
