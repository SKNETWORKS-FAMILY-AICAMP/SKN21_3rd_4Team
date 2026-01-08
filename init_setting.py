### docker init setting
# docker login
# docker pull luccia/skn21-3rd-4team
# docker run -p 6333:6333 -p 6334:6334 luccia/skn21-3rd-4team
###

### run init_setting.py
# python init_setting.py
###

from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from src.utils.config import ConfigDB

def init_qdrant():  
    # Client 초기화 - Qdrant 서버와 연결.
    client = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT)

    # Collection 생성 - DB/Table(Point(데이터) 저장공간)
    COLLECTION_NAME = ConfigDB.COLLECTION_NAME

    # 기존에 있던 Vector DB를 삭제 후 새로 만들기때문에 초기 설정에만 사용하도록.
    if client.collection_exists(collection_name=COLLECTION_NAME): # 있는지 여부 확인
        client.delete_collection(collection_name=COLLECTION_NAME) # 삭제

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=ConfigDB.VECTOR_SIZE,              
            distance=ConfigDB.DISTANCE_FUNCTION
        )
    )

    # 컬렉션 조회
    print(client.info())
    print(client.get_collections())
    print("docker와 Qdrant 서버 초기 설정 완료!!!")


def recover_snapshots():
    '''
    hugging face에서 snapshot 복구하기
    '''     
    # Hugging Face Snapshot URL (Raw/Resolve URL 사용)
    # blob -> resolve 로 변경하여 직접 다운로드 가능한 링크로 설정
    snapshot_url = "https://huggingface.co/datasets/lucymoon/skn21_3rd_4team/resolve/main/" + "learning_ai-5440725894880113-2026-01-07-08-14-04.snapshot"

    print(f"URL: {snapshot_url}")

    # Qdrant 클라이언트 연결 (타임아웃 10분 설정)
    client = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT, timeout=600)
    collection_name = ConfigDB.COLLECTION_NAME

    # 기존 컬렉션이 있으면 삭제
    if client.collection_exists(collection_name=collection_name):
        print(f"Removing existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    # URL을 통해 스냅샷 복구
    client.recover_snapshot(
        collection_name=collection_name,
        location=snapshot_url,
        wait=True
    )
    print(">>>> Snapshot recovery from Hugging Face completed successfully!")


def recover_quiz_snapshots():
    '''
    Hugging Face에서 퀴즈 스냅샷 복구하기
    '''
    # blob -> resolve 로 변경!
    snapshot_url = "https://huggingface.co/datasets/reasonableplan/skn21_3rd_4team/resolve/main/quizzes.snapshot"
    
    print(f"📂 퀴즈 스냅샷 URL: {snapshot_url}")
    
    client = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT, timeout=600)
    collection_name = "quizzes"
    
    # 기존 컬렉션이 있으면 삭제
    if client.collection_exists(collection_name=collection_name):
        print(f"⚠️ 기존 '{collection_name}' 컬렉션 삭제 중...")
        client.delete_collection(collection_name=collection_name)
    
    # URL을 통해 스냅샷 복구
    client.recover_snapshot(
        collection_name=collection_name,
        location=snapshot_url,
        wait=True
    )
    print("✅ 퀴즈 스냅샷 복구 완료!")


if __name__ == "__main__":
    # init_qdrant()
    recover_snapshots()
    recover_quiz_snapshots()