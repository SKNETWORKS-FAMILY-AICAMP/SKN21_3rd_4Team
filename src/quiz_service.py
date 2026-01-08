"""
🧩 퀴즈 서비스 - Qdrant DB 기반

Qdrant 'quizzes' 컬렉션에서 퀴즈 데이터를 조회합니다.
"""

import random
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.utils.config import ConfigDB


class QuizService:
    def __init__(self):
        """Qdrant 클라이언트 초기화"""
        self.client = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT)
        self.collection_name = "quizzes"
        self._check_collection()

    def _check_collection(self):
        """컬렉션 존재 여부 확인"""
        if not self.client.collection_exists(self.collection_name):
            print(f"⚠️ QuizService: '{self.collection_name}' 컬렉션이 없습니다. init_setting.py를 실행하세요.")

    def get_quizzes(self, category='all', count=5):
        """
        카테고리와 개수에 맞춰 랜덤 퀴즈 반환
        :param category: 'python', 'lecture', 'all'
        :param count: 반환할 문제 수
        """
        # 필터 조건 설정
        scroll_filter = None
        
        if category == 'python':
            scroll_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="python_doc"))]
            )
        elif category == 'lecture':
            scroll_filter = Filter(
                must_not=[FieldCondition(key="source", match=MatchValue(value="python_doc"))]
            )
        # 'all'인 경우 필터 없음
        
        try:
            # Qdrant에서 데이터 가져오기 (scroll API)
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1000,  # 충분히 크게
                with_payload=True
            )
            
            quizzes = [point.payload for point in results]
            
            # 랜덤 샘플링
            if len(quizzes) <= count:
                random.shuffle(quizzes)
                return quizzes
            return random.sample(quizzes, count)
            
        except Exception as e:
            print(f"❌ QuizService 오류: {e}")
            return []
