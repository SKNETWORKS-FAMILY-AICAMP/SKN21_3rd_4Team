# AI ingestion : Vector DB 구축
# 인공지능(AI) 시스템에서 데이터를 수집하여 AI 모델 학습이나 분석에 사용할 수 있는 중앙 저장소로 이동, 저장하는 과정

"""
src/ingestion_lectures.py

강의자료 기반 Vector DB를 구축하기 위한 수집(ingestion) 유틸리티 모듈.

본 모듈은 다음 로직을 코드 레벨에서 재사용 가능하도록 정리한 것이다.
- `scripts/build_vector_db.py`
- `notebooks/preprocess_lecture_notes.ipynb`

즉, 강의자료(.ipynb)를
→ 파싱하고
→ 의미 단위로 분할(chunking)한 뒤
→ 임베딩을 생성하여
→ Qdrant(Vector DB)에 업로드하고,
→ 스냅샷 생성 및 기본 검색 테스트까지 수행한다.

본 모듈은 스크립트/노트북과 달리
**프로그래밍 인터페이스(API)** 형태로 제공되어,
Agent, 서비스 코드, 파이프라인 등에서 직접 호출할 수 있다.

사용 예시:
    from src.ingestion_lectures import Ingestor
    ing = Ingestor()
    ing.build(provider='local')

CLI 실행:
    python -m src.ingestion_lectures \
        --provider local \
        --batch-size 128 \
        --run-retriever-test
"""

import os
import sys
import nbformat
import uuid
import logging
from pathlib import Path
from typing import List, Dict

# Ensure repo root on sys.path when executed as module
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.utils.config import ConfigDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Ingestor:
    def __init__(self, data_dir: str = 'data/raw/lectures', batch_size: int = 128):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

        # splitters
        self.header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')])
        self.code_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

        # Qdrant client (lazy init)
        self.client = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT)

    def parse_ipynb_cells(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        cells = []
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'markdown':
                cells.append({'type': 'markdown', 'content': cell.source, 'cell_index': idx})
            elif cell.cell_type == 'code':
                code_block = f"```python\n{cell.source}\n```"
                cells.append({'type': 'code', 'content': code_block, 'cell_index': idx})
        return cells

    def prepare_documents_from_cells(self, file_name: str, cells: List[Dict]) -> List[Dict]:
        docs = []
        lecture_title = os.path.splitext(file_name)[0]

        for cell in cells:
            base_meta = {
                'source_file': file_name,
                'lecture_title': lecture_title,
                'cell_index': cell['cell_index'],
                'cell_type': cell['type']
            }
            content = cell['content']

            heading = None
            if cell['type'] == 'markdown':
                lines = [ln for ln in content.splitlines() if ln.strip()]
                for ln in lines:
                    if ln.strip().startswith('#'):
                        heading = ln.strip().lstrip('#').strip()
                        break
                parts = self.header_splitter.split_text(content)
            else:
                parts = self.code_splitter.split_text(content)

            for i, part in enumerate(parts):
                meta = base_meta.copy()
                meta['chunk_id'] = i
                if heading:
                    meta['heading'] = heading
                meta['text_snippet'] = str(part)[:500]
                docs.append({'text': str(part), 'metadata': meta})
        return docs

    def get_embedding_provider(self, provider: str = 'local', model_name: str = 'text-embedding-3-small'):
        provider = provider.lower()
        return OpenAIEmbeddings(model=model_name)

    def upload_texts_to_qdrant(self, texts: List[str], metadatas: List[Dict], embedding_provider, collection_name: str = None):
        collection_name = collection_name or ConfigDB.COLLECTION_NAME

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_provider,
            validate_collection_config=False
        )

        ids = [str(uuid.uuid4()) for _ in texts]
        added = vector_store.add_texts(texts, metadatas=metadatas, ids=ids, batch_size=self.batch_size)
        logger.info('Uploaded %d texts to collection %s', len(added), collection_name)
        return added

    def create_snapshot_and_download(self, collection_name: str = None, out_dir: str = './snapshots') -> Path:
        collection_name = collection_name or ConfigDB.COLLECTION_NAME
        snap = self.client.create_snapshot(collection_name=collection_name)
        name = snap.name
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        download_url = f"http://{ConfigDB.HOST}:{ConfigDB.PORT}/collections/{collection_name}/snapshots/{name}"
        import requests
        r = requests.get(download_url)
        r.raise_for_status()
        p = out_dir / name
        with open(p, 'wb') as f:
            f.write(r.content)
        logger.info('Downloaded snapshot to %s', p)
        return p

    def simple_retriever_test(self, vector_store: QdrantVectorStore, queries: List[str]):
        for q in queries:
            docs = vector_store.similarity_search(q, k=5)
            print('\n=== QUERY:', q)
            for i, d in enumerate(docs[:5]):
                meta = getattr(d, 'metadata', {})
                snippet = getattr(d, 'page_content', '')[:200]
                print(f'[{i}] {meta.get("source_file")} | {meta.get("heading")} | {snippet.replace("\n"," ")[:200]}')

    def build(self, provider: str = 'local', local_model: str = 'text-embedding-3-small', openai_model: str = 'text-embedding-3-small', recreate_collection: bool = False, run_retriever_test: bool = False):
        if recreate_collection:
            try:
                self.client.delete_collection(collection_name=ConfigDB.COLLECTION_NAME)
                logger.info('Deleted existing collection %s', ConfigDB.COLLECTION_NAME)
            except Exception as e:
                logger.warning('Could not delete collection (it might not exist): %s', e)

        emb = self.get_embedding_provider(provider, local_model if provider == 'local' else openai_model)

        # parse and prepare
        all_docs = []
        for p in sorted(self.data_dir.glob('*.ipynb')):
            cells = self.parse_ipynb_cells(str(p))
            docs = self.prepare_documents_from_cells(p.name, cells)
            all_docs.extend(docs)

        texts = [d['text'] for d in all_docs]
        metadatas = [d['metadata'] for d in all_docs]

        # Optional OpenAI test
        if provider == 'openai' and hasattr(emb, 'embed_query'):
            try:
                emb.embed_query('test')
                logger.info('OpenAI embedding test successful')
            except Exception as e:
                logger.error('OpenAI embedding test failed: %s', e)
                raise

        # upload
        self.upload_texts_to_qdrant(texts=texts, metadatas=metadatas, embedding_provider=emb)

        # snapshot
        self.create_snapshot_and_download()

        # retriever test
        if run_retriever_test:
            vs = QdrantVectorStore(client=self.client, collection_name=ConfigDB.COLLECTION_NAME, embedding=emb, validate_collection_config=False)
            self.simple_retriever_test(vs, ['과적합이란 무엇인가?', '교차검증이란 무엇인가?', 'SVM이 언제 사용되는가?'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=['local', 'openai'], default=os.getenv('EMBEDDING_PROVIDER','local'))
    parser.add_argument('--local-model', default='text-embedding-3-small')
    parser.add_argument('--openai-model', default='text-embedding-3-small')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--recreate-collection', action='store_true')
    parser.add_argument('--run-retriever-test', action='store_true')
    args = parser.parse_args()

    ing = Ingestor(batch_size=args.batch_size)
    ing.build(provider=args.provider, local_model=args.local_model, openai_model=args.openai_model, recreate_collection=args.recreate_collection, run_retriever_test=args.run_retriever_test)
