import os
import sys
import glob
import requests
from pathlib import Path
from qdrant_client import QdrantClient
from src.utils.config import ConfigDB, ConfigAPI
from huggingface_hub import HfApi

def init_snapshots(folder_path: str):
    '''
    snapshots 폴더 초기화
    '''
    path = Path(folder_path)
    
    # snapshots 폴더가 있으면 내부 파일들 전체 삭제
    if path.exists():
        for file in path.glob("*"):
            file.unlink()

    # snapshots 폴더 생성 (있으면 pass)
    path.mkdir(parents=True, exist_ok=True)


def save_snapshots() -> str:
    '''
    스냅샷 생성 후 저장

    return: 스냅샷 파일 이름
    '''

    # 스냅샷 생성
    snapshot = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT).create_snapshot(collection_name=ConfigDB.COLLECTION_NAME)
    snapshot_name = snapshot.name
    print(snapshot_name)

    local_folder_path = "./data/snapshots/"

    # 스냅샷 폴더 초기화
    init_snapshots(local_folder_path)

    # 스냅샷 파일 다운로드 (서버 -> 로컬)
    download_url = f"http://{ConfigDB.HOST}:{ConfigDB.PORT}/collections/{ConfigDB.COLLECTION_NAME}/snapshots/{snapshot_name}"
    local_file_path = local_folder_path + snapshot_name  # 현재 디렉토리에 저장

    print(f"Downloading snapshot {snapshot_name} from {download_url}...")

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(local_file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f">>>> Downloaded to local: {os.path.abspath(local_file_path)}")

    return snapshot_name


def upload_snapshots(snapshot_name: str):
    '''
    스냅샷 파일 업로드
    '''
    # Hugging Face 업로드
    api = HfApi(token=ConfigAPI.HUGGINGFACE_API_KEY)
    repo_id = "lucymoon/skn21_3rd_4team"
    path_in_repo = snapshot_name
    local_file_path = "./data/snapshots/" + snapshot_name

    print(f"Uploading {local_file_path} to {repo_id}...")

    try:
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(">>>> Upload completed successfully!")
    except Exception as e:
        print(f">>>> Upload failed: {e}")


if __name__ == "__main__":
    name = save_snapshots()
    upload_snapshots(name)
