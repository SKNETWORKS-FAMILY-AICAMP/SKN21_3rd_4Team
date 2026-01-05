from typing import List, Dict, Any
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from src.utils.config import ConfigDB, ConfigAPI
from src.schema.search import SearchConfig



# SearchExecutor


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
        # 1. Qdrant 클라이언트 연결 (DB에 접속)

        self.client = QdrantClient(
            host=ConfigDB.HOST,
            port=ConfigDB.PORT
        )

        # 2. 임베딩 모델 설정 (질문을 벡터로 바꾸는 도구)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=ConfigAPI.OPENAI_API_KEY
        )

        # 3. 사용할 컬렉션 이름 (어느 방을 뒤질지)
        self.collection_name = ConfigDB.COLLECTION_NAME



    def execute_search(self, query: str, config: SearchConfig) -> List[Dict]:
        """
        Qdrant에서 실제 검색을 수행하는 함수
        
        Args:
            query (str): 검색할 질문 키워드 (예: "머신러닝이 뭐야?")
            config (SearchConfig): Role A가 준 검색 설정 (source, top_k 등)
            
        Returns:
            List[Dict]: 검색된 문서들의 리스트
        """

        try:

            # 1. 질문(텍스트)을 벡터(숫자)로 변환
            query_vector = self.embeddings.embed_query(query)

            # 2. 몇 개 가져올지 설정 (없으면 기본 5개)
            top_k = config.get("top_k", 5)

            # 3. Qdrant에서 검색 (query_points 사용 - 1.7+ 버전)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            
            # 4. 결과 정리
            results = []
            for hit in search_result.points:  # .points 추가!
                results.append({
                    "content": hit.payload.get('page_content', ''),
                    "score": hit.score,
                    "metadata": hit.payload.get('metadata', {})
                })

            return results

        except Exception as e:
            # 에러가 나면 멈추지 말고, 에러 메시지를 출력하고 빈 리스트를 줍니다.
            print(f"⚠️ [Executor] 검색 에러 발생: {e}")
            return []

        
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        중복 제거: 내용이 똑같은 게 여러 개 나오면 하나만 남깁니다.
        """
        seen_content = set()
        unique_results = []
        
        for res in results:
            content_sig = res['content'].strip()[:50]
            
            if content_sig in seen_content:
                continue
                
            seen_content.add(content_sig)
            unique_results.append(res)
            
        return unique_results
    def build_context(self, results: List[Dict]) -> str:
        """
        보고서 작성: LLM이 읽기 좋게 문장으로 정리합니다.
        """
        if not results:
            return "검색된 관련 자료가 없습니다."
        context_parts = []
        for i, res in enumerate(results, 1): # 번호는 1번부터
            source = res['metadata'].get('source', 'Unknown')
            score = round(res['score'], 2)
            content = res['content'].strip()
            
            part = f"[{i}] 출처: {source} (유사도: {score})\n{content}"
            context_parts.append(part)
            
        return "\n\n---\n\n".join(context_parts)



# 실행 명령어 python -m src.agent.nodes.search_executor


if __name__ == "__main__":
    # 1. 실행기(Executor) 생성
    executor = SearchExecutor()
    
    # 2. 테스트용 질문 & 설정 준비
    query = "머신러닝이 뭐야?"
    config = {
        "sources": ["lecture"],
        "top_k": 3
    }
    
    # 3. 검색 실행!
    print(f"🚀 테스트 시작: 질문 = '{query}'")
    results = executor.execute_search(query, config)
    
    # 4. 결과 정리 (중복 제거 & 보고서 작성)
    deduped = executor.deduplicate_results(results)
    context = executor.build_context(deduped)
    
    # 5. 최종 결과 출력
    print("\n✅ 정리된 검색 결과:")
    print(context)
            
            
            