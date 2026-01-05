# AI ingestion : Vector DB 구축
# 인공지능(AI) 시스템에서 데이터를 수집하여 AI 모델 학습이나 분석에 사용할 수 있는 중앙 저장소로 이동, 저장하는 과정

import os
from typing import List, Dict, Any, Optional

class Ingestor:
    def __init__(
        self,
        docs_root: str,
        include_dirs: Optional[List[str]] = None,
        include_exts: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "python_docs",
        embedding_model_name: str = "text-embedding-3-small",
        batch_size: int = 64,
    ):
        self.docs_root = docs_root
        self.include_dirs = include_dirs or ["library", "reference"]
        self.include_exts = include_exts or [".txt"]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size

    def load_repo(self, repo_url: str) -> str:
        # 1) 로컬 경로면 그대로 반환
        # 2) zip이면 해제 후 경로 반환
        # 3) git url이면 clone 후 경로 반환(선택)
        pass

    def collect_files(self, root_path: str) -> List[str]:
        # include_dirs + include_exts 기반 파일 리스트 수집
        pass

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        # {"content": str, "metadata": dict} 반환
        # metadata 예: {"source":"python_doc", "title":"library/pathlib.txt"}
        pass

    def split_text(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        # [{"content": chunk_text, "metadata": {...}}] 반환
        # python docs면 reST 헤딩 기반 split → 길이 split
        pass

    def upload_to_qdrant(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        # embed + upsert (배치)
        # return {"uploaded": n, "failed": m}
        pass

    def run(self, repo_url: Optional[str] = None) -> Dict[str, int]:
        root_path = self.load_repo(repo_url) if repo_url else self.docs_root
        file_paths = self.collect_files(root_path)

        total_uploaded, total_failed = 0, 0
        for fp in file_paths:
            parsed = self.parse_file(fp)
            chunks = self.split_text(parsed)
            stats = self.upload_to_qdrant(chunks)
            total_uploaded += stats.get("uploaded", 0)
            total_failed += stats.get("failed", 0)

        return {"uploaded": total_uploaded, "failed": total_failed}

if __name__ == "__main__":
    ingestor = Ingestor(
        docs_root="../data/raw/python-3.14-docs-text",
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="python_doc_embed_small",
    )
    stats = ingestor.run()
    print(stats)