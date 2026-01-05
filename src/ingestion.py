# AI ingestion : Vector DB 구축
# 인공지능(AI) 시스템에서 데이터를 수집하여 AI 모델 학습이나 분석에 사용할 수 있는 중앙 저장소로 이동, 저장하는 과정

import os
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


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
        load_dotenv(override=True)

        self.docs_root = os.path.abspath(docs_root)
        self.include_dirs = include_dirs or ["library", "reference"]
        self.include_exts = include_exts or [".txt"]

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size

        self._vector_store: Optional[QdrantVectorStore] = None

    # -------------------------
    # 0) 내부 유틸
    # -------------------------
    @staticmethod
    def _split_rst_sections(text: str) -> List[Tuple[str, str]]:
        """
        reST 스타일 섹션 분리:
        제목 줄 + underline(====, ---- 등) 패턴 기준
        """
        lines = text.splitlines()
        sections: List[Tuple[str, str]] = []
        current_title = "ROOT"
        buf: List[str] = []

        def flush():
            nonlocal buf, current_title
            content = "\n".join(buf).strip()
            if content:
                sections.append((current_title, content))
            buf = []

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if i + 1 < len(lines):
                underline = lines[i + 1].rstrip()
                if underline and len(underline) >= max(3, len(line)):
                    # underline이 같은 문자로만 구성되고, reST 섹션에 자주 쓰는 문자면 섹션으로 판단
                    if len(set(underline)) == 1 and underline[0] in "= -~^*+_":
                        flush()
                        current_title = line.strip() or current_title
                        i += 2
                        continue
            buf.append(lines[i])
            i += 1

        flush()
        return sections

    def _get_vector_store(self) -> QdrantVectorStore:
        if self._vector_store is not None:
            return self._vector_store

        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        embedding = OpenAIEmbeddings(model=self.embedding_model_name)
        
        # 컬렉션이 없으면 생성
        if not client.collection_exists(collection_name=self.collection_name):
            # embedding 모델에 따른 벡터 크기 결정
            vector_size = 1536 if "3-small" in self.embedding_model_name else 1536
            
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"컬렉션 '{self.collection_name}' 생성 완료!")

        # QdrantVectorStore 생성
        # validate_collection_config=False: 컬렉션을 직접 생성했으므로 검증 불필요
        self._vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=embedding,
            validate_collection_config=False,
        )
        return self._vector_store

    # -------------------------
    # 1) Repo / Data 준비
    # -------------------------
    def load_repo(self, repo_url: str) -> str:
        """
        - 로컬 경로면 그대로 반환
        - .zip이면 같은 폴더에 압축 해제 후 폴더 경로 반환
        (git clone은 현재 단계에서 제외)
        """
        if not repo_url:
            raise ValueError("repo_url이 비어있습니다.")

        p = Path(repo_url)

        # 로컬 디렉토리
        if p.exists() and p.is_dir():
            return str(p.resolve())

        # 로컬 zip
        if p.exists() and p.is_file() and p.suffix.lower() == ".zip":
            out_dir = p.with_suffix("")  # xxx.zip -> xxx
            out_dir = out_dir.resolve()
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(str(p), "r") as zf:
                    zf.extractall(str(out_dir))
            return str(out_dir)

        raise FileNotFoundError(f"repo_url 경로를 찾을 수 없습니다: {repo_url}")

    def collect_files(self, root_path: str) -> List[str]:
        """
        include_dirs 하위에서 include_exts 확장자만 수집
        """
        root = Path(root_path)
        if not root.exists():
            raise FileNotFoundError(f"root_path가 존재하지 않습니다: {root_path}")

        targets: List[str] = []
        exts = {e.lower() for e in self.include_exts}

        for d in self.include_dirs:
            base = root / d
            if not base.exists():
                # 범위를 최소로 시작하므로, 없는 디렉토리는 그냥 스킵
                continue
            for fp in base.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in exts:
                    targets.append(str(fp.resolve()))

        return targets

    # -------------------------
    # 2) Parse / Split
    # -------------------------
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        파일 1개를 읽어서 {"content": str, "metadata": dict} 형태로 반환
        전처리는 '최소'로: reST 구조(제목 underline 등) 깨지지 않게 유지
        """
        fp = Path(file_path)
        if not fp.exists() or not fp.is_file():
            raise FileNotFoundError(f"file_path가 유효하지 않습니다: {file_path}")

        # docs_root 기준 상대경로를 title로 사용
        rel_path = os.path.relpath(str(fp), self.docs_root).replace(os.sep, "/")

        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # 최소 정리: null 문자 정도만 제거 (과도한 특수문자 제거 금지)
        text = text.replace("\x00", "")

        return {
            "content": text,
            "metadata": {
                "source": "python_doc",
                "title": rel_path,
            },
        }

    def split_text(self, parsed: Dict[str, Any]) -> List[Document]:
        """
        1) reST 섹션 단위 분리
        2) 섹션 문서를 chunk_size / overlap으로 재분할
        최종 산출물: List[Document]
        """
        content: str = parsed["content"]
        base_meta: Dict[str, Any] = parsed["metadata"]

        section_docs: List[Document] = []
        for section_title, section_text in self._split_rst_sections(content):
            section_docs.append(
                Document(
                    page_content=section_text,
                    metadata={
                        **base_meta,
                        "section": section_title,
                    },
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # langchain splitter는 Document 리스트를 받아 chunk Document 리스트를 반환
        chunk_docs = splitter.split_documents(section_docs)

        # chunk index 부여(디버깅/추적용)
        for idx, d in enumerate(chunk_docs):
            d.metadata["chunk_index"] = idx

        return chunk_docs

    # -------------------------
    # 3) Upload
    # -------------------------
    def upload_to_qdrant(self, chunks: List[Document]) -> Dict[str, int]:
        """
        VectorStore.add_documents로 업로드 (배치)
        """
        vector_store = self._get_vector_store()

        uploaded = 0
        failed = 0

        # 배치 업로드
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            try:
                # langchain-qdrant는 ids 반환(버전에 따라 다를 수 있음)
                vector_store.add_documents(batch)
                uploaded += len(batch)
            except Exception:
                failed += len(batch)

        return {"uploaded": uploaded, "failed": failed}

    # -------------------------
    # 4) Run
    # -------------------------
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
    # 스크립트 파일 위치 기준으로 경로 설정
    script_dir = Path(__file__).parent  # src/
    project_root = script_dir.parent    # 프로젝트 루트
    docs_path = project_root / "data" / "raw" / "python-3.14-docs-text"
    
    ingestor = Ingestor(
        docs_root=str(docs_path),
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="python_doc_embed_small",
        embedding_model_name="text-embedding-3-small",
        batch_size=64,
    )
    stats = ingestor.run()
    print(stats)