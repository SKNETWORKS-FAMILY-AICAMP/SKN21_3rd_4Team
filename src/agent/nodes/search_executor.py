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
        """
        try:
            # 1. 질문(텍스트)을 벡터(숫자)로 변환
            query_vector = self.embeddings.embed_query(query)
            # 2. 몇 개 가져올지 설정 (없으면 기본 5개)
            top_k = config.get("top_k", 5)
            # [수정된 부분] search_method 확인
            method = config.get('search_method', 'similarity')
            if method == 'mmr':
                print("ℹ️ MMR 검색 요청됨 (현재는 기본 검색으로 동작)")
            # 3. Qdrant에서 검색 (query_points 사용 - 1.7+ 버전)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            
            # 4. 결과 정리
            results = []
            for hit in search_result.points:
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

    def prepare_for_analysis_agent(self, query: str, results: List[Dict], config: dict) -> dict:
        """
        Analysis Agent에게 넘길 형식으로 변환
        
        Args:
            query: 원본 질문
            results: 검색 결과 리스트 (deduplicate 후)
            config: Router가 생성한 검색 설정
            
        Returns:
            Analysis Agent가 기대하는 JSON 형식
        """
        return {
            "query": query,                           # 원본 질문
            "retrieved_documents": [                  # 검색된 문서 리스트
                {
                    "content": r['content'],          # 문서 내용
                    "metadata": {
                        "source": r['metadata'].get('source', 'unknown'),     # 출처
                        "title": r['metadata'].get('title', 'Unknown'),       # 파일명
                        "page": r['metadata'].get('page', None),              # 페이지 번호
                        "chunk_index": r['metadata'].get('chunk_index', None) # 조각 번호
                    },
                    "score": round(r['score'], 4)     # 유사도 점수
                }
                for r in results
            ],
            "search_metadata": {                      # 검색 정보
                "total_found": len(results),
                "sources_searched": config.get('sources', []),
                "search_method": config.get('search_method', 'similarity')
            }
        }






# 실행 명령어 python -m src.agent.nodes.search_executor


if __name__ == "__main__":
    """
    Search Executor 단독 테스트
    Router가 생성하는 config와 유사한 설정으로 테스트
    """
    executor = SearchExecutor()
    
    # Router와 동일한 테스트 질문들 (search_router.py 참고)
    test_cases = [
        {
            "query": "RAG가 뭐야?",
            "config": {"sources": ["lecture"], "top_k": 3, "search_method": "similarity"}
        },
        {
            "query": "Python list comprehension 문법",
            "config": {"sources": ["python_doc"], "top_k": 3, "search_method": "similarity"}
        },
        {
            "query": "딥러닝 모델 최적화 방법",
            "config": {"sources": ["lecture"], "top_k": 7, "search_method": "mmr"}
        }
    ]
    
    print("=" * 60)
    print("🧪 Search Executor 단독 테스트 (Router 설정 시뮬레이션)")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        config = case["config"]
        
        print(f"\n� [{i}] 질문: {query}")
        print(f"   설정: {config}")
        print("-" * 60)
        
        # 검색 실행
        results = executor.execute_search(query, config)
        deduped = executor.deduplicate_results(results)
        context = executor.build_context(deduped)
        
        # 결과 요약 (전체 context 말고 첫 200자만)
        preview = context[:200] + "..." if len(context) > 200 else context
        print(f"   => {len(deduped)}개 문서 검색됨")
        print(f"   => 첫 번째 결과 미리보기:\n{preview}")
        print("=" * 60)