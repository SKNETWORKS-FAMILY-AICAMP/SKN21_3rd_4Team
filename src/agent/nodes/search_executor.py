# SearchExecutor

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from src.utils.config import ConfigDB, ConfigAPI


class SearchExecutor:
    """
    검색 실행 에이전트

    Router가 정해준 설정(Config)에 따라
    실제 Vector DB(Qdrant)를 조회하고 결과를 반환합니다.
    """

    def __init__(self):
        """
        초기화 메서드: DB 연결과 임베딩 모델을 준비합니다.
        """
        # Qdrant 클라이언트 연결
        self.client = QdrantClient(
            host=ConfigDB.HOST,
            port=ConfigDB.PORT
        )

        # 임베딩 모델 설정
        self.embeddings = OpenAIEmbeddings(
            model=ConfigDB.EMBEDDING_MODEL,
            api_key=ConfigAPI.OPENAI_API_KEY
        )

        # 사용할 컬렉션 이름
        self.collection_name = ConfigDB.COLLECTION_NAME

    def build_context(self, results: List[Dict]) -> str:
        """
        LLM이 읽기 좋게 문장으로 정리합니다.
        """
        if not results:
            return "검색된 관련 자료가 없습니다."
        
        context_parts = []
        for i, res in enumerate(results, 1):
            source = res['metadata'].get('source', 'Unknown')
            score = round(res['score'], 2)
            content = res['content'].strip()
            
            part = f"[{i}] 출처: {source} (유사도: {score})\n{content}"
            context_parts.append(part)
            
        return "\n\n---\n\n".join(context_parts)

