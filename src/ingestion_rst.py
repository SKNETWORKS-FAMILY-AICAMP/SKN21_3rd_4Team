# RST 파일 전용 Ingestion 테스트 스크립트
# 유사도 최적화를 위한 설정 적용

import os
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# ============================================================
# 테스트 설정 (여기서 쉽게 변경 가능)
# ============================================================
# 임베딩 모델 선택: "text-embedding-3-small" 또는 "text-embedding-3-large"
EMBEDDING_MODEL = "text-embedding-3-large"  # ← 여기 변경!

# 컬렉션 이름 (None이면 기본값 "learning_ai" 사용)
COLLECTION_NAME = "learning_ai"  # ← 필요시 변경

# 컬렉션 재생성 여부 (기존 컬렉션 삭제 후 재생성)
RECREATE_COLLECTION = False  # ← True로 설정하면 기존 컬렉션 삭제 후 재생성


class RSTIngestor:
    def __init__(
        self,
        # 청크 설정: 유사도 최적화
        chunk_size: int = 900,
        chunk_overlap: int = 200,
        # Qdrant
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "learning_ai",
        recreate_collection: bool = False,
        # Embedding
        # 주의: lecture와 같은 컬렉션을 사용하면 같은 임베딩 모델을 사용해야 함
        embedding_model_name: str = None,  # None이면 파일 상단 EMBEDDING_MODEL 사용
        batch_size: int = 32,
    ):
        load_dotenv(override=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.recreate_collection = recreate_collection
        # 파일 상단 EMBEDDING_MODEL 사용 (None이면)
        self.embedding_model_name = embedding_model_name if embedding_model_name is not None else EMBEDDING_MODEL
        self.batch_size = batch_size

        self._vector_store: Optional[QdrantVectorStore] = None

    # -------------------------
    # 0) RST 정제 및 파싱 유틸
    # -------------------------
    @staticmethod
    def _guess_vector_size(embedding_model_name: str) -> int:
        """
        OpenAI 임베딩 모델별 벡터 차원 추정값.
        필요 시 컬렉션을 새로 만들 때 사용.
        """
        # 참고: text-embedding-3-small=1536, text-embedding-3-large=3072
        known = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return known.get(embedding_model_name, 1536)

    @staticmethod
    def _clean_inline_markup(text: str) -> str:
        """
        제목/헤더 등에 들어가는 RST 인라인 마크업을 최소한으로 정리.
        - :role:`...` -> ...
        - `text`_ -> text
        - ``code`` 는 유지
        """
        import re

        def replace_role(match):
            content = match.group(1)
            content = re.sub(r"\s*<[^>]+>\s*", "", content)  # "list <...>" -> "list"
            content = content.lstrip("!~")
            return content

        text = re.sub(r":[a-zA-Z0-9_\-\.\+!]+:`([^`]+)`", replace_role, text)
        text = re.sub(r"`([^`]+)`_", r"\1", text)
        return text.strip()

    @staticmethod
    def _clean_rst_noise(text: str) -> str:
        """
        RST 문법 노이즈를 강력하게 제거
        라인 단위로 처리하여 directive를 통째로 제거
        
        강화된 제거 대상:
        - 한 줄짜리 directive (.. highlight::, .. _label:)
        - .. index:: 블록
        - 버전 관련 directive (versionadded, versionchanged, deprecated, availability)
        - RST role (:func:`...`, :class:`...` 등)
        - 제목 장식 문자 (===, ---, ~~~, ^^^)
        """
        import re
        
        cleaned_lines = []
        skip_until_blank = False
        skip_index_block = False
        in_code_block = False
        code_base_indent: Optional[int] = None
        
        for line in text.splitlines():
            stripped = line.strip()

            # -------------------------
            # 0) code-block / doctest 블록: 내용은 최대한 유지
            # -------------------------
            if in_code_block:
                if stripped == "":
                    cleaned_lines.append("")
                    # 빈 줄은 코드 블록을 끝내지 않음 (RST에서 코드 블록 내부 빈 줄 허용)
                    continue

                # 들여쓰기가 줄어들면 코드 블록 종료로 판단
                cur_indent = len(line) - len(line.lstrip(" \t"))
                if code_base_indent is not None and cur_indent < code_base_indent:
                    in_code_block = False
                    code_base_indent = None
                    # 아래 로직으로 다시 처리
                else:
                    # 코드 본문은 deindent해서 저장 (검색 키워드 보존)
                    if code_base_indent is None:
                        code_base_indent = cur_indent
                    cleaned_lines.append(line[code_base_indent:])
                    continue
            
            # === 1. 스킵 대상 한 줄짜리 directive ===
            if stripped.startswith(('.. highlight::', '.. _', '.. versionadded::', '.. versionchanged::', 
                                    '.. deprecated::', '.. availability::', '.. seealso::', '.. rubric::',
                                    '.. sectionauthor::')):
                continue
                
            # === 2. index/contents 블록 처리 ===
            # 목차나 인덱스는 검색에 도움 안 됨
            if stripped.startswith(('.. index::', '.. toctree::', '.. contents::', '.. include::')):
                skip_index_block = True
                continue
            
            if skip_index_block:
                if stripped == '' or (line.startswith(' ') or line.startswith('\t')):
                    if stripped == '':
                        skip_index_block = False
                    continue
                else:
                    skip_index_block = False
            
            # === 3. 일반 directive 블록 처리 ===
            if stripped.startswith('.. '):
                # 3-1) 코드 블록 directive: 본문은 유지
                code_directives = (".. code-block::", ".. code::", ".. sourcecode::", ".. doctest::")
                if stripped.startswith(code_directives):
                    # 언어 힌트가 있으면 1줄 남겨주기 (검색/요약에 도움)
                    m = re.match(r"^\.\.\s+([a-zA-Z0-9_\-:]+)::\s*(.*)$", stripped)
                    if m:
                        dname, arg = m.group(1), m.group(2).strip()
                        if arg:
                            cleaned_lines.append(f"{dname} {arg}")
                    in_code_block = True
                    code_base_indent = None
                    continue

                # 3-2) API/구조 directive: '시그니처 라인'은 반드시 남긴다
                # 예) ".. class:: PurePath(*pathsegments)" -> "class PurePath(*pathsegments)"
                keep_signature = [
                    "py:module", "py:class", "py:function", "py:method", "py:attribute",
                    "module", "class", "function", "method", "exception", "data", "attribute",
                    "c:function", "c:type", "c:var", "c:macro", "c:member",
                ]
                m = re.match(r"^\.\.\s+([a-zA-Z0-9_\-:]+)::\s*(.*)$", stripped)
                if m and m.group(1) in keep_signature:
                    dname = m.group(1)
                    arg = m.group(2).strip()
                    sig_line = f"{dname} {arg}".strip()
                    if sig_line:
                        cleaned_lines.append(sig_line)
                    skip_until_blank = False
                    continue

                # 3-3) 그 외 directive는 블록 전체 스킵 (이미지/toctree 등)
                skip_until_blank = True
                continue
            
            # directive 블록 내부면 스킵
            if skip_until_blank:
                if stripped == '':
                    skip_until_blank = False
                    cleaned_lines.append('')
                continue

            # === 3.5 directive 옵션 라인 제거 (":synopsis:" 같은 것) ===
            # 보통 directive 아래에 붙는 옵션들은 검색 품질에 도움 적음
            if (line.startswith(' ') or line.startswith('\t')) and stripped.startswith(':') and stripped.endswith(':') is False:
                # 예: "   :synopsis: ..." / "   :align: center"
                if re.match(r"^:[a-zA-Z0-9_\-]+:\s*", stripped):
                    continue
            
            # === 4. 제목 장식 문자 라인 제거 ===
            if stripped and len(stripped) >= 3:
                if len(set(stripped)) == 1 and stripped[0] in '=-~^*+#':
                    continue
            
            # === 5. RST role 치환 (향상됨) ===
            # 패턴: :role:`content` 또는 :role:`!content` 또는 :role:`~content`
            # Role 이름에는 알파벳, 숫자, _, -, ., + 허용
            # Content에는 <Link> 가 있을 수 있음 -> Link 텍스트 제거하고 앞부분만 남김
            
            def replace_role(match):
                content = match.group(1)
                # 1. <Link> 제거 (예: list <typesseq-list> -> list)
                content = re.sub(r'\s*<[^>]+>\s*', '', content)
                # 2. 선행 !, ~ 제거 (예: ~list.append -> list.append)
                content = content.lstrip('!~')
                return content

            # Role Regex: :role:`...`
            # role 이름에 !, ., - 포함 가능
            line = re.sub(r':[a-zA-Z0-9_\-\.\+!]+:`([^`]+)`', replace_role, line)
            
            # :option:`--flag` 형태 등
            line = re.sub(r':option:`([^`]+)`', r'\1', line)
            
            # 외부 링크 마커 제거 `text`_  -> text
            line = re.sub(r'`([^`]+)`_', r'\1', line)
            
            # 중복 공백 정리 (탭 -> 공백, 다중 공백 -> 단일 공백)
            # 단, 들여쓰기는 유지
            if not line.startswith(' ') and not line.startswith('\t'):
                line = re.sub(r'[ \t]+', ' ', line)
            
            cleaned_lines.append(line)
        
        # 결과 조합
        result = '\n'.join(cleaned_lines)
        
        # === 7. 최종 정리 ===
        # 과도한 빈 줄 정리 (3개 이상 -> 2개)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # 시작/끝 공백 제거
        return result.strip()
    
    @staticmethod
    def _parse_rst_sections(text: str) -> List[Tuple[str, str, str]]:
        """
        RST 섹션을 계층적으로 파싱
        반환: [(h1_title, h2_title, content), ...]
        """
        lines = text.splitlines()
        sections: List[Tuple[str, str, str]] = []
        
        current_h1 = "ROOT"
        current_h2 = ""
        buf: List[str] = []

        def flush():
            nonlocal buf, current_h1, current_h2
            content = "\n".join(buf).strip()
            if content:
                sections.append((current_h1, current_h2, content))
            buf = []

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # underline 체크
            if i + 1 < len(lines):
                underline = lines[i + 1].rstrip()
                if underline and len(underline) >= max(3, len(line)):
                    if len(set(underline)) == 1:
                        char = underline[0]
                        
                        # H1: ===== (최상위)
                        if char == '=':
                            flush()
                            current_h1 = RSTIngestor._clean_inline_markup(line.strip()) or current_h1
                            current_h2 = ""
                            i += 2
                            continue
                        
                        # H2: ----- (하위)
                        elif char == '-':
                            flush()
                            current_h2 = RSTIngestor._clean_inline_markup(line.strip())
                            i += 2
                            continue
                        
                        # H3 이하: ~~~~, ^^^^, ++++
                        elif char in '~^+*_':
                            # H3는 현재 섹션에 포함
                            buf.append(line)
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

        # 컬렉션 없으면 생성
        expected_vector_size = self._guess_vector_size(self.embedding_model_name)
        if self.recreate_collection and client.collection_exists(collection_name=self.collection_name):
            client.delete_collection(collection_name=self.collection_name)
            print(f"[OK] collection deleted: '{self.collection_name}'")

        if not client.collection_exists(collection_name=self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=expected_vector_size, distance=Distance.COSINE),
            )
            print(f"[OK] collection created: '{self.collection_name}' (vector_size={expected_vector_size})")
        else:
            # 기존 컬렉션과 임베딩 모델 차원이 다르면 업로드가 실패할 수 있어 경고
            try:
                info = client.get_collection(self.collection_name)
                actual_size = None
                if getattr(info, "config", None) and getattr(info.config, "params", None):
                    vectors = getattr(info.config.params, "vectors", None)
                    if getattr(vectors, "size", None) is not None:
                        actual_size = vectors.size
                if actual_size and actual_size != expected_vector_size:
                    print(
                        f"[WARN] collection vector_size mismatch: expected={expected_vector_size}, actual={actual_size}. "
                        f"Use a new collection for embedding_model='{self.embedding_model_name}'."
                    )
            except Exception:
                pass

        self._vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=embedding,
            validate_collection_config=False,
        )
        return self._vector_store

    # -------------------------
    # 1) Parse / Split
    # -------------------------
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        RST 파일 1개를 읽어서 원문+메타데이터 반환
        (섹션 파싱은 원문 기반으로 수행해야 제목 구조가 보존됨)
        """
        fp = Path(file_path)
        if not fp.exists() or not fp.is_file():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        # 파일명을 title로
        file_name = fp.stem  # introduction

        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # 1. null 문자 제거
        text = text.replace("\x00", "")

        return {
            "content": text,
            "metadata": {
                "source": "python_doc_rst",
                "title": file_name,
                "file_path": str(fp),
            },
        }

    def split_text(self, parsed: Dict[str, Any]) -> List[Document]:
        """
        RST 섹션 단위 분리 → chunk 분할
        """
        content: str = parsed["content"]
        base_meta: Dict[str, Any] = parsed["metadata"]

        section_docs: List[Document] = []
        
        # RST 섹션 파싱 (원문 기반)
        for h1, h2, section_text_raw in self._parse_rst_sections(content):
            # 섹션 본문 정제 (제목 구조는 메타/프리픽스로 보존)
            section_text = self._clean_rst_noise(section_text_raw)

            # 코드 블록 포함 여부 체크 (정제 후 기준)
            has_code = ("::" in section_text) or ("code-block" in section_text) or ("doctest" in section_text)
            h1_clean = h1.strip() if h1 else "ROOT"
            h2_clean = h2.strip() if h2 else ""
            
            section_docs.append(
                Document(
                    page_content=section_text,
                    metadata={
                        **base_meta,
                        "section": h1_clean,
                        "subsection": h2_clean if h2_clean else h1_clean,
                        "has_code": has_code,
                    },
                )
            )

        # RST 특화 splitter: 구분자 우선순위 명시
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",       # 단락 구분 (최우선)
                "\n::",       # RST 코드 블록
                "\n.. ",      # RST directive
                "\n",         # 일반 줄바꿈
                " ",          # 공백
                ""            # 마지막 수단
            ]
        )

        # chunk 분할
        chunk_docs = splitter.split_documents(section_docs)

        # chunk index 부여
        for idx, d in enumerate(chunk_docs):
            d.metadata["chunk_index"] = idx
            # 모든 chunk에 문맥 프리픽스를 붙여 검색 품질 향상
            prefix_lines = [
                f"[TITLE] {d.metadata.get('title', '')}",
                f"[H1] {d.metadata.get('section', '')}",
            ]
            if d.metadata.get("subsection") and d.metadata.get("subsection") != d.metadata.get("section"):
                prefix_lines.append(f"[H2] {d.metadata.get('subsection', '')}")
            prefix = "\n".join(prefix_lines).strip()
            if prefix:
                d.page_content = prefix + "\n\n" + d.page_content
            # 첫 200자를 snippet으로
            d.metadata["snippet"] = d.page_content[:200].replace("\n", " ")

        return chunk_docs

    # -------------------------
    # 2) Upload
    # -------------------------
    def upload_to_qdrant(self, chunks: List[Document]) -> Dict[str, int]:
        """
        VectorStore에 업로드 (배치)
        """
        vector_store = self._get_vector_store()

        uploaded = 0
        failed = 0

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            try:
                # 재실행 시 중복 적재를 줄이기 위한 deterministic id
                ids = []
                for d in batch:
                    raw = f"{d.metadata.get('file_path','')}/{d.metadata.get('section','')}/{d.metadata.get('subsection','')}/{d.metadata.get('chunk_index','')}"
                    ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, raw)))
                vector_store.add_documents(batch, ids=ids)
                uploaded += len(batch)
                # print 제거 (run_all에서 로그 출력)
            except Exception as e:
                print(f"  [ERR] batch {i//self.batch_size + 1} failed: {e}")
                failed += len(batch)

        return {"uploaded": uploaded, "failed": failed}

    # -------------------------
    # 3) Run
    # -------------------------
    def run(self, file_path: str, verbose: bool = True) -> Dict[str, Any]:
        """
        단일 파일 ingestion
        """
        if verbose:
            print(f"[LOAD] {file_path}")
        parsed = self.parse_file(file_path)
        
        if verbose:
            print(f"[SPLIT] chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        chunks = self.split_text(parsed)
        
        if verbose:
            print(f"[INFO] total_chunks={len(chunks)}")
        
            # 샘플 출력
            print("\n--- Sample chunks (first 3) ---")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n[chunk {i+1}]")
                print(f"  section: {chunk.metadata.get('section', 'N/A')}")
                print(f"  subsection: {chunk.metadata.get('subsection', 'N/A')}")
                print(f"  has_code: {chunk.metadata.get('has_code', False)}")
                print(f"  preview: {chunk.metadata.get('snippet', '')[:120]}...")
            print("---\n")
        
        if verbose:
            print("[UPLOAD] Qdrant 업로드 중...")
        stats = self.upload_to_qdrant(chunks)
        
        return {
            **stats,
            "total_chunks": len(chunks),
            "file_path": str(file_path),
        }

    def run_all(self, directory: str, verbose: bool = True) -> Dict[str, Any]:
        """
        디렉토리 내 모든 .rst 파일 재귀 ingestion
        """
        import glob
        
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"디렉토리를 찾을 수 없습니다: {directory}")
        
        # 모든 .rst 파일 찾기 (재귀)
        rst_files = list(dir_path.rglob("*.rst"))
        
        print("=" * 60)
        print(f"[DIR] RST Ingestion: {directory}")
        print(f"   Found .rst files: {len(rst_files)}")
        print("=" * 60)
        
        total_stats = {
            "total_files": len(rst_files),
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "uploaded": 0,
            "failed": 0,
            "errors": [],
        }
        
        for idx, rst_file in enumerate(rst_files, 1):
            rel_path = rst_file.relative_to(dir_path)
            
            # 불필요한 파일 스킵
            if rst_file.name in ['genindex.rst', 'py-modindex.rst', 'search.rst', 'contents.rst', 'copyright.rst', 'license.rst']:
                print(f"\n[{idx}/{len(rst_files)}] {rel_path} (SKIPPED: Generated/Meta file)")
                continue
                
            print(f"\n[{idx}/{len(rst_files)}] {rel_path}")
            
            try:
                stats = self.run(str(rst_file), verbose=False)
                total_stats["processed_files"] += 1
                total_stats["total_chunks"] += stats["total_chunks"]
                total_stats["uploaded"] += stats["uploaded"]
                total_stats["failed"] += stats["failed"]
                print(f"  [OK] chunks={stats['total_chunks']}, uploaded={stats['uploaded']}")
            except Exception as e:
                total_stats["failed_files"] += 1
                total_stats["errors"].append({"file": str(rel_path), "error": str(e)})
                print(f"  [FAIL] {e}")
        
        print("\n" + "=" * 60)
        print("=== Ingestion Summary ===")
        print("=" * 60)
        print(f"  total_files: {total_stats['total_files']}")
        print(f"  processed_files: {total_stats['processed_files']}")
        print(f"  failed_files: {total_stats['failed_files']}")
        print(f"  total_chunks: {total_stats['total_chunks']}")
        print(f"  uploaded: {total_stats['uploaded']}")
        print(f"  failed_uploads: {total_stats['failed']}")
        
        if total_stats["errors"]:
            print("\n[!] Failed files:")
            for err in total_stats["errors"][:10]:  # 최대 10개만 출력
                print(f"  - {err['file']}: {err['error'][:50]}")
        
        return total_stats




if __name__ == "__main__":
    import argparse
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    rst_dir = project_root / "data" / "raw" / "python_rst"
    
    parser = argparse.ArgumentParser(description="RST Ingestion")
    parser.add_argument("--single", action="store_true", help="Single file only (default: all files)")
    parser.add_argument("--file", type=str, help="Single file path")
    parser.add_argument("--collection", type=str, default=None, help=f"Qdrant collection name (None이면 파일 상단 COLLECTION_NAME={COLLECTION_NAME} 사용)")
    parser.add_argument("--recreate-collection", action="store_true", default=None, help="Delete and recreate collection before upload (None이면 파일 상단 RECREATE_COLLECTION 사용)")
    parser.add_argument("--embedding-model", type=str, default=None, help=f"OpenAI embedding model (None이면 파일 상단 EMBEDDING_MODEL={EMBEDDING_MODEL} 사용)")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Upload 없이 청킹 결과만 출력")
    args = parser.parse_args()

    print("=" * 60)
    print("RST Ingestion (optimized chunking)")
    print("=" * 60)

    # 파일 상단 설정 사용 (명령줄 인자가 None이면)
    collection_name = args.collection if args.collection is not None else COLLECTION_NAME
    recreate_collection = args.recreate_collection if args.recreate_collection is not None else RECREATE_COLLECTION
    embedding_model = args.embedding_model if args.embedding_model is not None else EMBEDDING_MODEL
    
    ingestor = RSTIngestor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name=collection_name,
        recreate_collection=recreate_collection,
        embedding_model_name=embedding_model,
        batch_size=args.batch_size,
    )

    if args.single or args.file:
        # 단일 파일 ingestion
        test_file = Path(args.file) if args.file else rst_dir / "introduction.rst"
        if not test_file.exists():
            print(f"[ERR] File not found: {test_file}")
            exit(1)
        if args.dry_run:
            parsed = ingestor.parse_file(str(test_file))
            chunks = ingestor.split_text(parsed)
            print(f"\n[DRY RUN] chunks={len(chunks)}")
            for i, c in enumerate(chunks[:3], 1):
                print(f"\n--- chunk {i} ---")
                print(f"section={c.metadata.get('section')} subsection={c.metadata.get('subsection')} has_code={c.metadata.get('has_code')}")
                print(c.page_content[:500])
            stats = {"total_chunks": len(chunks), "uploaded": 0, "failed": 0}
        else:
            stats = ingestor.run(str(test_file))
        
        print("\n" + "=" * 60)
        print("[DONE]")
        print(f"  - Total chunks: {stats['total_chunks']}")
        print(f"  - Uploaded: {stats['uploaded']}")
        print(f"  - Failed: {stats['failed']}")
        print("=" * 60)
    else:
        # 기본: 전체 디렉토리 ingestion
        if not rst_dir.exists():
            print(f"[ERR] Directory not found: {rst_dir}")
            exit(1)
        if args.dry_run:
            # 디렉토리 dry-run은 비용이 클 수 있어 안내만
            print("[DRY RUN] 디렉토리 전체는 업로드 없이 실행하지 않습니다. --single/--file로 확인하세요.")
        else:
            stats = ingestor.run_all(str(rst_dir))
