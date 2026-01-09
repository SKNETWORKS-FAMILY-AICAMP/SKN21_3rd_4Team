"""
강의 자료 Ingestion 모듈

이 모듈의 역할:
- 강의자료(.ipynb)를 파싱하여 chunk 단위로 분할합니다
- OpenAI Embedding으로 벡터화한 뒤 Qdrant(Vector DB)에 저장합니다
- source 메타데이터로 'lecture'를 부여하여 python_doc과 구분합니다
- 전처리 로직을 적용하여 이미지/URL/HTML/LaTeX를 제거합니다
- Qdrant 컬렉션이 없으면 자동으로 생성합니다

핵심 개념: 문맥 주입 (Context Injection)
- 각 chunk 앞에 강의 제목과 섹션 정보를 추가하여 검색 품질 향상
- Markdown과 Code를 별도 처리하여 최적화된 chunk 생성
- 헤더 기반 분할 후 RecursiveCharacterTextSplitter로 세부 분할
"""

import os
import re
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
from utils.config import ConfigDB
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


class Ingestor:
    # 한국어 불용어 (검색 품질 향상용)
    STOPWORDS = {'이', '그', '저', '것', '등', '및', '또한', '그리고', '하지만', '그러나', '따라서', '그래서', '즉', '예를', '들어'}
    
    def __init__(
        self,
        docs_root: str,
        include_exts: Optional[List[str]] = None,
        # chunk 옵션
        md_chunk_size: int = 1200,        
        md_chunk_overlap: int = 200,     
        code_chunk_size: int = 1000,      
        code_chunk_overlap: int = 150,
        # Qdrant
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "learning_ai",
        # Embedding
        embedding_model_name: str = ConfigDB.EMBEDDING_MODEL,
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

    def _get_vector_store(self) -> QdrantVectorStore:
        if self._vector_store is not None:
            return self._vector_store

        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        embedding = OpenAIEmbeddings(model=self.embedding_model_name)

        if not client.collection_exists(collection_name=self.collection_name):
            vector_size = ConfigDB.VECTOR_SIZE

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

    def _preprocess_markdown(self, text: str) -> str:
        """마크다운 텍스트 전처리: 불필요한 요소 제거"""
        # 1) Base64 인코딩된 이미지 데이터 제거
        text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '', text)
        
        # 2) 이미지 마크다운 문법 완전 제거 (![alt](url) 형식)
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        
        # 3) URL을 '[링크]'로 대체
        text = re.sub(r'https?://[^\s\)\]]+', '[링크]', text)
        
        # 4) HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        
        # 5) LaTeX 수식: $ 기호만 제거하고 내용 유지
        # $x_3 = x_1^2$ → x_3 = x_1^2
        # $$수식$$ → 수식
        text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)  # $$...$$ 처리
        text = re.sub(r'\$([^$]+)\$', r'\1', text)       # $...$ 처리
        
        # 6) 마크다운 헤더 # 기호 제거 (### 제목 → 제목)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # 7) 불필요한 공백/개행 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

    def _preprocess_code(self, code: str) -> str:
        """코드 텍스트 전처리: 주석 정리 및 핵심 코드 유지"""
        # 1) 빈 줄이 과도하게 많은 경우 정리
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        # 2) 여러 개의 # 주석을 단일 #으로 정규화 (### 주석 → # 주석)
        code = re.sub(r'^(\s*)#{2,}\s*', r'\1# ', code, flags=re.MULTILINE)
        
        return code.strip()

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
                # ✅ 전처리 적용
                txt = self._preprocess_markdown(txt)
                
                # ✅ 추가 필터링: 10자 이하이거나 '[링크]'가 포함된 경우 제외
                # 의미 없는 텍스트나 링크만 있는 셀 제거
                if len(txt.strip()) <= 10 or "[링크]" in txt:
                    continue

                if txt.strip():
                    cells.append(
                        {"cell_type": "markdown", "text": txt, "cell_index": idx}
                    )
            elif cell.cell_type == "code":
                code = cell.source or ""
                # ✅ 코드 전처리 적용
                code = self._preprocess_code(code)
                
                # ✅ 너무 짧은 코드 셀은 건너뛰기 (검색에 도움 안 됨)
                # 예: rfc.feature_importances_ 같은 한 줄짜리 셀
                if len(code.strip()) < 30:
                    continue
                    
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

    def _build_context_prefix(self, meta: Dict[str, Any]) -> str:
        """
        문맥 주입: 청크 앞에 강의 제목과 섹션 정보를 추가
        검색 시 "머신러닝" 같은 키워드가 [강의: 머신러닝]과 매칭되어 정확도 향상
        """
        parts = []
        
        # 강의 제목 추가
        lecture_title = meta.get("lecture_title", "")
        if lecture_title:
            parts.append(f"[강의: {lecture_title}]")
        
        # 섹션(헤더) 정보 추가 (H1, H2, H3 순서대로)
        headers = []
        for h in ["H1", "H2", "H3"]:
            if h in meta and meta[h]:
                headers.append(meta[h])
        if headers:
            parts.append(f"[섹션: {' > '.join(headers)}]")
        
        if parts:
            return "\n".join(parts) + "\n\n"
        return ""

    def split_text(self, parsed: Dict[str, Any]) -> List[Document]:
        """
        셀 단위 → chunk(Document) 리스트 생성
        - markdown: header 기반 split 후(큰 덩어리) 추가로 recursive chunk
        - code: recursive chunk
        - ✅ 문맥 주입: 각 청크 앞에 강의 제목과 섹션 정보 추가
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
                        # ✅ 문맥 주입
                        context_prefix = self._build_context_prefix(meta)
                        enriched_chunk = context_prefix + chunk
                        
                        out_docs.append(
                            Document(
                                page_content=enriched_chunk,
                                metadata={
                                    **meta,
                                    "text_snippet": chunk[:200],  # 원본 청크 스니펫 유지
                                },
                            )
                        )
            else:
                # code chunk
                for chunk in self.code_splitter.split_text(cell_text):
                    # ✅ 코드에도 문맥 주입 (어떤 강의의 코드인지 알 수 있음)
                    context_prefix = self._build_context_prefix(cell_meta)
                    enriched_chunk = context_prefix + chunk
                    
                    out_docs.append(
                        Document(
                            page_content=enriched_chunk,
                            metadata={
                                **cell_meta,
                                "text_snippet": chunk[:200],  # 원본 청크 스니펫 유지
                            },
                        )
                    )

        # 추적용 chunk_index
        for i, d in enumerate(out_docs):
            d.metadata["chunk_index"] = i

        return out_docs

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
    project_root = script_dir.parent        # 프로젝트 루트
    lectures_path = project_root / "data" / "raw" / "lectures"

    ingestor = Ingestor(
        docs_root=str(lectures_path),
        qdrant_host=ConfigDB.HOST,
        qdrant_port=int(ConfigDB.PORT),
        collection_name=ConfigDB.COLLECTION_NAME,
        # embedding_model_name은 ConfigDB.EMBEDDING_MODEL 기본값 사용
        batch_size=64,
    )
    stats = ingestor.run()
    print(stats)