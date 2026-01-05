"""
build_vector_db

Purpose:
    강의자료(.ipynb) 노트들을 파싱하여
    의미 기반 검색이 가능한 Vector DB(Qdrant)를 구축한다.

Features:
    - ipynb 파일에서 markdown / code 셀 파싱
    - 셀 단위 chunking 및 메타데이터 보강
    - 로컬 / OpenAI 임베딩 선택 가능
    - Qdrant Vector DB 업로드 (batch 처리)
    - 컬렉션 스냅샷 생성 및 다운로드

Args:
    ...

Returns:
    ...
"""

import os
import sys
import argparse
import nbformat
import logging
import time
import uuid
from pathlib import Path
from typing import List

# Ensure repo root is on path
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

from src.utils.config import ConfigDB
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_vector_db")


def parse_ipynb_cells(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    cells = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            text = cell.source
            cells.append({'type': 'markdown', 'content': text, 'cell_index': idx})
        elif cell.cell_type == 'code':
            code = cell.source
            code_block = f"```python\n{code}\n```"
            cells.append({'type': 'code', 'content': code_block, 'cell_index': idx})
    return cells


def prepare_documents_from_cells(file_name: str, cells: List[dict]):
    docs = []
    lecture_title = os.path.splitext(file_name)[0]

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')])
    code_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

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
            parts = header_splitter.split_text(content)
        else:
            parts = code_splitter.split_text(content)

        for i, part in enumerate(parts):
            meta = base_meta.copy()
            meta['chunk_id'] = i
            if heading:
                meta['heading'] = heading
            meta['text_snippet'] = str(part)[:500]
            docs.append({'text': str(part), 'metadata': meta})
    return docs


class LocalHFEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if SentenceTransformer is None:
            raise RuntimeError('sentence-transformers is not available')
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        embs = self.model.encode(texts, show_progress_bar=False)
        return [list(map(float, e)) for e in embs]

    def embed_query(self, text: str):
        e = self.model.encode([text], show_progress_bar=False)[0]
        return list(map(float, e))


def get_embedding_provider(provider_name: str, model_name: str):
    provider_name = provider_name.lower()
    if provider_name == 'openai':
        if OpenAIEmbeddings is None:
            raise RuntimeError('OpenAI embeddings provider not available (langchain_openai missing)')
        logger.info('Using OpenAI embeddings model=%s', model_name)
        emb = OpenAIEmbeddings(model=model_name)
        return emb
    else:
        logger.info('Using local SentenceTransformers embeddings model=%s', model_name)
        return LocalHFEmbeddings(model_name)


def upload_texts_to_qdrant(client: QdrantClient, collection_name: str, texts: List[str], metadatas: List[dict], batch_size=256, embedding_provider=None):
    # Wrap embedding provider into a LangChain-compatible object when needed by QdrantVectorStore
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_provider,
        validate_collection_config=False
    )

    # Generate ids
    ids = [str(uuid.uuid4()) for _ in texts]

    logger.info('Uploading %d texts to Qdrant in batches of %d', len(texts), batch_size)
    added_ids = vector_store.add_texts(texts, metadatas=metadatas, ids=ids, batch_size=batch_size)
    logger.info('Added ids: %d', len(added_ids))
    return added_ids


def create_snapshot_and_download(client: QdrantClient, collection_name: str, out_dir='./snapshots'):
    snap = client.create_snapshot(collection_name=collection_name)
    name = snap.name
    logger.info('Created snapshot: %s', name)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    download_url = f"http://{ConfigDB.HOST}:{ConfigDB.PORT}/collections/{collection_name}/snapshots/{name}"
    import requests
    r = requests.get(download_url)
    r.raise_for_status()
    p = Path(out_dir) / name
    with open(p, 'wb') as f:
        f.write(r.content)
    logger.info('Downloaded snapshot to: %s', p)
    return p


def simple_retriever_test(vector_store: QdrantVectorStore, queries: List[str], k=5):
    for q in queries:
        docs = vector_store.similarity_search(q, k=k)
        print('\n=== QUERY:', q)
        for i, d in enumerate(docs[:k]):
            meta = getattr(d, 'metadata', {})
            snippet = getattr(d, 'page_content', '')[:200]
            print(f'[{i}] {meta.get("source_file")} | {meta.get("heading")} | {snippet.replace("\n"," ")[:200]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=['local', 'openai'], default=os.getenv('EMBEDDING_PROVIDER','local'))
    parser.add_argument('--openai-model', default='text-embedding-3-small')
    parser.add_argument('--local-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--recreate-collection', action='store_true')
    parser.add_argument('--data-dir', default='data/raw/lectures')
    parser.add_argument('--run-retriever-test', action='store_true')
    args = parser.parse_args()

    provider = args.provider
    model_name = args.openai_model if provider == 'openai' else args.local_model

    client = QdrantClient(host=ConfigDB.HOST, port=ConfigDB.PORT)

    if args.recreate_collection:
        try:
            client.delete_collection(collection_name=ConfigDB.COLLECTION_NAME)
            logger.info('Deleted existing collection: %s', ConfigDB.COLLECTION_NAME)
        except Exception as e:
            logger.warning('Could not delete collection (may not exist): %s', e)

    emb = get_embedding_provider(provider, model_name)

    # Parse and prepare docs
    all_docs = []
    data_dir = Path(args.data_dir)
    for p in sorted(data_dir.glob('*.ipynb')):
        cells = parse_ipynb_cells(str(p))
        docs = prepare_documents_from_cells(p.name, cells)
        all_docs.extend(docs)

    texts = [d['text'] for d in all_docs]
    metadatas = [d['metadata'] for d in all_docs]

    # Optional: test OpenAI key by performing a single embedding
    if provider == 'openai':
        try:
            emb.embed_query('test')
            logger.info('OpenAI embedding test successful')
        except Exception as e:
            logger.error('OpenAI embedding test failed: %s', e)
            raise

    upload_texts_to_qdrant(client=client, collection_name=ConfigDB.COLLECTION_NAME, texts=texts, metadatas=metadatas, batch_size=args.batch_size, embedding_provider=emb)

    # snapshot and download
    create_snapshot_and_download(client, ConfigDB.COLLECTION_NAME)

    # retriever test
    if args.run_retriever_test:
        vs = QdrantVectorStore(client=client, collection_name=ConfigDB.COLLECTION_NAME, embedding=emb, validate_collection_config=False)
        simple_retriever_test(vs, ['과적합이란 무엇인가?', '교차검증이란 무엇인가?', 'SVM이 언제 사용되는가?'])


if __name__ == '__main__':
    main()
