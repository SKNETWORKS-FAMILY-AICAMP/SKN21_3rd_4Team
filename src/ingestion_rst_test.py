# RST 파일 전용 Ingestion 테스트 스크립트
# 유사도 최적화를 위한 설정 적용

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


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
        # Embedding
        embedding_model_name: str = "text-embedding-3-small",
        batch_size: int = 32,
    ):
        load_dotenv(override=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size

        self._vector_store: Optional[QdrantVectorStore] = None

    # -------------------------
    # 0) RST 정제 및 파싱 유틸
    # -------------------------
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
        
        for line in text.splitlines():
            stripped = line.strip()
            
            # === 1. 스킵 대상 한 줄짜리 directive ===
            # highlight directive (코드 하이라이팅 설정)
            if stripped.startswith('.. highlight::'):
                continue
            
            # 참조 레이블 (.. _label-name:)
            if stripped.startswith('.. _') and stripped.endswith(':'):
                continue
            
            # 버전 정보 directive (한 줄)
            if stripped.startswith(('.. versionadded::', '.. versionchanged::', 
                                     '.. deprecated::', '.. availability::')):
                continue
            
            # seealso 참조 (한 줄)
            if stripped.startswith('.. seealso::'):
                continue
                
            # === 2. index 블록 처리 ===
            if stripped.startswith('.. index::'):
                # index는 블록일 수도 있고 한 줄일 수도 있음
                skip_index_block = True
                continue
            
            if skip_index_block:
                # 들여쓰기 있으면 index 블록 계속
                if stripped == '' or (line.startswith(' ') or line.startswith('\t')):
                    if stripped == '':
                        skip_index_block = False
                    continue
                else:
                    skip_index_block = False
                    # 현재 라인은 처리 계속
                    
            # === 3. 일반 directive 블록 처리 ===
            if stripped.startswith('.. '):
                # 특정 directive는 내용 유지 (c:function, c:type 등 API 정의)
                if any(stripped.startswith(f'.. {d}::') for d in 
                       ['c:function', 'c:type', 'c:var', 'c:macro', 'c:member',
                        'py:function', 'py:class', 'py:method', 'py:attribute',
                        'note', 'warning', 'tip', 'important', 'caution']):
                    # 이 directive들은 마커만 제거하고 내용은 유지
                    # 마커 라인은 스킵하지만 다음 내용은 유지
                    skip_until_blank = False
                    continue
                else:
                    # 그 외 directive는 블록 전체 스킵
                    skip_until_blank = True
                    continue
            
            # directive 블록 내부면 스킵
            if skip_until_blank:
                if stripped == '':
                    skip_until_blank = False
                    cleaned_lines.append('')  # 빈 줄은 유지
                continue
            
            # === 4. 제목 장식 문자 라인 제거 ===
            # 전체가 같은 문자로만 구성된 라인 (===, ---, ~~~, ^^^, ***)
            if stripped and len(stripped) >= 3:
                if len(set(stripped)) == 1 and stripped[0] in '=-~^*+#':
                    continue
            
            # === 5. RST role 치환 ===
            # :role:`text` -> text (c:func, py:class, ref, doc, pep 등)
            line = re.sub(r':[a-zA-Z0-9_~]+:`([^`]+)`', r'\1', line)
            
            # :option:`--flag` 형태도 처리
            line = re.sub(r':option:`([^`]+)`', r'\1', line)
            
            # === 6. 기타 RST 문법 정리 ===
            # 주석 참조 제거 [#]_
            line = re.sub(r'\[#\]_', '', line)
            
            # RST 코드 마커를 일반 따옴표로
            line = line.replace('``', '"')
            
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
                            current_h1 = line.strip() or current_h1
                            current_h2 = ""
                            i += 2
                            continue
                        
                        # H2: ----- (하위)
                        elif char == '-':
                            flush()
                            current_h2 = line.strip()
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
        if not client.collection_exists(collection_name=self.collection_name):
            vector_size = 1536  # text-embedding-3-small

            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"✅ 컬렉션 '{self.collection_name}' 생성 완료! (vector_size={vector_size})")

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
        RST 파일 1개를 읽어서 강력하게 정제 후 반환
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
        
        # 2. RST  노이즈 강력 제거 (개선!)
        text = self._clean_rst_noise(text)

        return {
            "content": text,
            "metadata": {
                "source": "python_doc_rst",
                "title": file_name,
            },
        }

    def split_text(self, parsed: Dict[str, Any]) -> List[Document]:
        """
        RST 섹션 단위 분리 → chunk 분할
        """
        content: str = parsed["content"]
        base_meta: Dict[str, Any] = parsed["metadata"]

        section_docs: List[Document] = []
        
        # RST 섹션 파싱
        for h1, h2, section_text in self._parse_rst_sections(content):
            # 코드 블록 포함 여부 체크
            has_code = "::" in section_text
            
            section_docs.append(
                Document(
                    page_content=section_text,
                    metadata={
                        **base_meta,
                        "section": h1,
                        "subsection": h2 if h2 else h1,
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
                vector_store.add_documents(batch)
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
            print(f"📄 파일 로딩: {file_path}")
        parsed = self.parse_file(file_path)
        
        if verbose:
            print(f"✂️  청킹 중... (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        chunks = self.split_text(parsed)
        
        if verbose:
            print(f"📊 총 {len(chunks)}개 청크 생성")
        
            # 샘플 출력
            print("\n--- 샘플 청크 (처음 3개) ---")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n[청크 {i+1}]")
                print(f"  섹션: {chunk.metadata.get('section', 'N/A')}")
                print(f"  하위섹션: {chunk.metadata.get('subsection', 'N/A')}")
                print(f"  코드 포함: {chunk.metadata.get('has_code', False)}")
                print(f"  미리보기: {chunk.metadata.get('snippet', '')[:100]}...")
            print("---\n")
        
        if verbose:
            print(f"🚀 Qdrant 업로드 중...")
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
        print(f"  총 파일: {total_stats['total_files']}개")
        print(f"  처리 성공: {total_stats['processed_files']}개")
        print(f"  처리 실패: {total_stats['failed_files']}개")
        print(f"  총 청크: {total_stats['total_chunks']}개")
        print(f"  업로드 성공: {total_stats['uploaded']}개")
        print(f"  업로드 실패: {total_stats['failed']}개")
        
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
    parser.add_argument("--collection", type=str, default="learning_ai", help="Qdrant collection name")
    args = parser.parse_args()

    print("=" * 60)
    print("RST Ingestion (optimized chunking)")
    print("=" * 60)

    ingestor = RSTIngestor(
        chunk_size=900,
        chunk_overlap=200,
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name=args.collection,
        embedding_model_name="text-embedding-3-small",
        batch_size=32,
    )

    if args.single or args.file:
        # 단일 파일 ingestion
        test_file = Path(args.file) if args.file else rst_dir / "introduction.rst"
        if not test_file.exists():
            print(f"[ERR] File not found: {test_file}")
            exit(1)
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
        stats = ingestor.run_all(str(rst_dir))
