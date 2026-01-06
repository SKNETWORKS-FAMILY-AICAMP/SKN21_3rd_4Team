from typing import List, Dict
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from src.utils.config import ConfigDB, ConfigAPI
from src.schema.search import SearchConfig

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

        # 3. 사용할 컬렉션 이름
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


    def prepare_for_analysis_agent(self, query: str, results: List[Dict], config: dict) -> dict:
        """
        Analysis Agent에게 넘길 형식으로 변환
        
        Args:
            query: 원본 질문
            results: 검색 결과 리스트 (deduplicate 후)
            config: Router가 생성한 검색 설정 (top_k 포함)
            
        Returns:
            Analysis Agent가 기대하는 JSON 형식
        """
        # ✅ LLM이 결정한 top_k만큼만 결과를 자름 (명시적 적용)
        top_k = config.get('top_k', 5)
        limited_results = results[:top_k]
        
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
                for r in limited_results  # ✅ top_k 적용된 리스트 사용
            ],
            "search_metadata": {                      # 검색 정보
                "total_found": len(limited_results),  # ✅ 실제 전달되는 개수
                "top_k_requested": top_k,             # ✅ 요청된 top_k 값
                "sources_searched": config.get('sources', []),
                "search_method": config.get('search_method', 'similarity')
            }
        }