# AI ingestion : Vector DB 구축
# 인공지능(AI) 시스템에서 데이터를 수집하여 AI 모델 학습이나 분석에 사용할 수 있는 중앙 저장소로 이동, 저장하는 과정

"""
src/ingestion_lectures.py

강의자료(.ipynb)를 파싱하여 chunk 단위로 분할하고,
OpenAI Embedding으로 벡터화한 뒤 Qdrant(Vector DB)에 저장하는 ingestion 모듈.

- source 메타데이터로 'lecture'를 부여하여 python_doc과 구분 가능
- Qdrant 컬렉션이 없으면 자동 생성
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import nbformat
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


class Ingestor:
    def __init__(
        self,
        docs_root: str,
        include_exts: Optional[List[str]] = None,
        # chunk 옵션
        md_chunk_size: int = 1000,
        md_chunk_overlap: int = 100,
        code_chunk_size: int = 800,
        code_chunk_overlap: int = 50,
        # Qdrant
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "learning_ai_legacy",
        # Embedding
        embedding_model_name: str = "text-embedding-3-small",
        # Upload
        batch_size: int = 64,
    ):
        load_dotenv(override=True)

        self.docs_root = os.path.abspath(docs_root)
        self.include_exts = include_exts or [".ipynb"]

        self.md_chunk_size = md_chunk_size
        self.md_chunk_overlap = md_chunk_overlap
        self.code_chunk_size = code_chunk_size
        self.code_chunk_overlap = code_chunk_overlap

        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size

        # splitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
        )
        self.md_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.md_chunk_size,
            chunk_overlap=self.md_chunk_overlap,
        )
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.code_chunk_size,
            chunk_overlap=self.code_chunk_overlap,
        )

        self._vector_store: Optional[QdrantVectorStore] = None

    # -------------------------
    # 0) VectorStore 준비
    # -------------------------
    def _get_vector_store(self) -> QdrantVectorStore:
        if self._vector_store is not None:
            return self._vector_store

        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        embedding = OpenAIEmbeddings(model=self.embedding_model_name)

        # 컬렉션 없으면 생성
        if not client.collection_exists(collection_name=self.collection_name):
            # text-embedding-3-small = 1536 차원
            # (필요하면 모델명에 따라 분기 가능)
            vector_size = 1536

            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"컬렉션 '{self.collection_name}' 생성 완료! (vector_size={vector_size})")

        self._vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=embedding,
            validate_collection_config=False,
        )
        return self._vector_store

    # -------------------------
    # 1) Data 준비
    # -------------------------
    def load_repo(self, repo_url: str) -> str:
        """
        강의자료는 로컬 폴더(docs_root)를 그대로 사용하므로
        repo_url은 현재 단계에서는 '로컬 경로'만 지원(확장 가능).
        """
        if not repo_url:
            raise ValueError("repo_url이 비어있습니다.")
        p = Path(repo_url)
        if p.exists() and p.is_dir():
            return str(p.resolve())
        raise FileNotFoundError(f"repo_url 경로를 찾을 수 없습니다: {repo_url}")

    def collect_files(self, root_path: str) -> List[str]:
        """
        docs_root 하위에서 include_exts(.ipynb)만 수집
        """
        root = Path(root_path)
        if not root.exists():
            raise FileNotFoundError(f"root_path가 존재하지 않습니다: {root_path}")

        exts = {e.lower() for e in self.include_exts}
        targets: List[str] = []
        for fp in root.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in exts:
                targets.append(str(fp.resolve()))
        return sorted(targets)

    # -------------------------
    # 2) Parse / Split
    # -------------------------
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        ipynb 파일 1개를 읽어서 셀 단위로 추출:
        반환:
          {
            "cells": [{"cell_type": "markdown|code", "text": "...", "cell_index": n}, ...],
            "metadata": {...}
          }
        """
        fp = Path(file_path)
        if not fp.exists() or not fp.is_file():
            raise FileNotFoundError(f"file_path가 유효하지 않습니다: {file_path}")

        # 파일명 기반 title
        file_name = fp.name
        lecture_title = fp.stem

        nb = nbformat.read(str(fp), as_version=4)

        cells = []
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                txt = cell.source or ""
                if txt.strip():
                    cells.append(
                        {"cell_type": "markdown", "text": txt, "cell_index": idx}
                    )
            elif cell.cell_type == "code":
                code = cell.source or ""
                # code는 fenced block으로 감싸면 검색 시 문맥이 좋아짐
                code_block = f"```python\n{code}\n```"
                if code.strip():
                    cells.append(
                        {"cell_type": "code", "text": code_block, "cell_index": idx}
                    )

        return {
            "cells": cells,
            "metadata": {
                "source": "lecture",          # ✅ 핵심: 소스 구분
                "source_file": file_name,
                "lecture_title": lecture_title,
            },
        }

    def split_text(self, parsed: Dict[str, Any]) -> List[Document]:
        """
        셀 단위 → chunk(Document) 리스트 생성
        - markdown: header 기반 split 후(큰 덩어리) 추가로 recursive chunk
        - code: recursive chunk
        """
        base_meta: Dict[str, Any] = parsed["metadata"]
        cells: List[Dict[str, Any]] = parsed["cells"]

        out_docs: List[Document] = []

        for cell in cells:
            cell_type = cell["cell_type"]
            cell_text = cell["text"]
            cell_index = cell["cell_index"]

            # 공통 메타
            cell_meta = {
                **base_meta,
                "cell_type": cell_type,
                "cell_index": cell_index,
            }

            if cell_type == "markdown":
                # 1) header split
                header_parts = self.header_splitter.split_text(cell_text)

                # 2) header 파트 각각을 다시 chunking
                for part in header_parts:
                    # part는 Document 형태로 들어올 수 있음
                    if isinstance(part, Document):
                        text = part.page_content
                        meta = {**cell_meta, **(part.metadata or {})}
                    else:
                        text = str(part)
                        meta = dict(cell_meta)

                    for chunk in self.md_splitter.split_text(text):
                        out_docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    **meta,
                                    "text_snippet": chunk[:200],
                                },
                            )
                        )
            else:
                # code chunk
                for chunk in self.code_splitter.split_text(cell_text):
                    out_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **cell_meta,
                                "text_snippet": chunk[:200],
                            },
                        )
                    )

        # 추적용 chunk_index
        for i, d in enumerate(out_docs):
            d.metadata["chunk_index"] = i

        return out_docs

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

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            try:
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
    # 프로젝트 루트 기준: data/raw/lectures
    script_dir = Path(__file__).parent      # src/
    project_root = script_dir.parent.parent.parent        # 프로젝트 루트
    lectures_path = project_root / "data" / "raw" / "lectures"

    ingestor = Ingestor(
        docs_root=str(lectures_path),
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="learning_ai_legacy",
        embedding_model_name="text-embedding-3-small",
        batch_size=64,
    )
    stats = ingestor.run()
    print(stats)