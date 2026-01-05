"""
RAG 청킹 전략 자동 평가 스크립트

사용법:
    python scripts/evaluate_chunking.py
    
기능:
1. 다양한 chunk_size/overlap 조합으로 ingestion 실행
2. 각 설정마다 eval_dataset.py의 질문으로 검색 테스트
3. 정확도(top_k 안에 정답 파일 포함 여부) 측정
4. 결과를 CSV와 그래프로 저장
"""

import sys
import os
from pathlib import Path
import time
from typing import Dict, List, Any
import json
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.eval_dataset import EVALUATION_DATASET
from src.ingestion_lectures import Ingestor
from src.agent.nodes.search_router import build_search_config
from src.agent.nodes.search_executor import SearchExecutor


class ChunkingEvaluator:
    """청킹 전략별 성능 평가 클래스 (learning_ai 컬렉션 재사용)"""
    
    def __init__(self, collection_name: str = "learning_ai"):
        self.collection_name = collection_name  # 고정: learning_ai
        self.executor = SearchExecutor()
        self.results_dir = project_root / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def run_ingestion(self, config: Dict[str, int]) -> str:
        """
        특정 청킹 설정으로 learning_ai 컬렉션 재생성
        ✅ Python documentation (.txt) + 강의자료 (.ipynb) 모두 포함
        
        Args:
            config: {"md_chunk_size": 800, "md_chunk_overlap": 100, ...}
        
        Returns:
            collection_name (항상 "learning_ai")
        """
        print(f"\n{'='*70}")
        print(f"🔄 Ingestion 시작: {self.collection_name}")
        print(f"{'='*70}")
        print(f"  Markdown Chunk: size={config['md_chunk_size']}, overlap={config['md_chunk_overlap']}")
        print(f"  Code Chunk: size={config['code_chunk_size']}, overlap={config['code_chunk_overlap']}")
        
        # 경로 설정
        lectures_path = project_root / "data" / "raw" / "lectures"
        python_docs_path = project_root / "data" / "raw" / "python-3.14-docs-text"
        
        # learning_ai 컬렉션 삭제 (있으면)
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        
        try:
            if client.collection_exists(self.collection_name):
                client.delete_collection(self.collection_name)
                print(f"  🗑️  기존 {self.collection_name} 컬렉션 삭제")
        except Exception as e:
            print(f"  ⚠️  컬렉션 삭제 실패 (무시): {e}")
        
        # 1️⃣ Python documentation ingestion
        print(f"\n  📚 Python Documentation 업로드 중...")
        from src.ingestion import Ingestor as PythonIngestor
        
        python_ingestor = PythonIngestor(
            docs_root=str(python_docs_path),
            chunk_size=config["md_chunk_size"],  # txt도 md_chunk_size 사용
            chunk_overlap=config["md_chunk_overlap"],
            collection_name=self.collection_name,
            batch_size=64,
        )
        
        start_time = time.time()
        python_stats = python_ingestor.run()
        python_time = time.time() - start_time
        print(f"     ✅ Python docs: {python_stats['uploaded']}개 ({python_time:.1f}초)")
        
        # 2️⃣ 강의자료 ingestion (같은 컬렉션에 추가)
        print(f"\n  📝 강의자료 업로드 중...")
        
        lecture_ingestor = Ingestor(
            docs_root=str(lectures_path),
            md_chunk_size=config["md_chunk_size"],
            md_chunk_overlap=config["md_chunk_overlap"],
            code_chunk_size=config["code_chunk_size"],
            code_chunk_overlap=config["code_chunk_overlap"],
            collection_name=self.collection_name,  # 같은 컬렉션에 추가
            batch_size=64,
        )
        
        start_time = time.time()
        lecture_stats = lecture_ingestor.run()
        lecture_time = time.time() - start_time
        print(f"     ✅ Lectures: {lecture_stats['uploaded']}개 ({lecture_time:.1f}초)")
        
        total_uploaded = python_stats['uploaded'] + lecture_stats['uploaded']
        total_time = python_time + lecture_time
        print(f"\n  ✅ 전체 완료: {total_uploaded}개 업로드 ({total_time:.1f}초)")
        
        return self.collection_name
    
    def evaluate_collection(
        self, 
        test_questions: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        learning_ai 컬렉션에 대해 평가 데이터셋으로 검색 정확도 측정
        
        Returns:
            {
                "accuracy": 0.85,
                "correct": 18,
                "total": 21,
                "details": [...]
            }
        """
        print(f"\n{'='*70}")
        print(f"📊 평가 시작: {self.collection_name}")
        print(f"{'='*70}")
        
        correct = 0
        total = len(test_questions)
        details = []
        
        # 임시로 SearchExecutor의 컬렉션 이름 변경 (원래 코드 수정 필요 시)
        # 현재 SearchExecutor는 하드코딩된 컬렉션 이름 사용할 수 있음
        # 여기서는 간단히 전역으로 수정 (실제로는 리팩토링 필요)
        
        for i, qa in enumerate(test_questions, 1):
            question = qa["question"]
            expected_files = qa["expected_files"]
            
            try:
                # Router로 검색 설정 생성
                config = build_search_config(question)
                config["top_k"] = top_k  # 강제 설정
                
                # Executor로 검색 (collection_name을 어떻게 전달할지는 구현에 따라)
                # 현재 SearchExecutor가 고정 컬렉션 사용 중이므로, 
                # 이 부분은 실제 구현에 맞춰 수정해야 함
                
                # 직접 Qdrant 검색
                from qdrant_client import QdrantClient
                from langchain_openai import OpenAIEmbeddings
                
                client = QdrantClient(host="localhost", port=6333)
                embedding = OpenAIEmbeddings(model="text-embedding-3-small")
                
                query_vector = embedding.embed_query(question)
                
                # ✅ query_points 사용 (Qdrant 1.7+)
                search_results = client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                )
                
                # 검색된 파일명 추출
                retrieved_files = []
                for point in search_results.points:
                    # 메타데이터 구조에 따라 파일명 추출
                    metadata = point.payload.get("metadata", {})
                    source_file = (
                        point.payload.get("source_file") or
                        metadata.get("source_file") or
                        metadata.get("title") or
                        ""
                    )
                    if source_file:
                        retrieved_files.append(source_file)
                
                # 정답 확인
                is_correct = any(
                    expected in retrieved_files[0] if retrieved_files else False
                    for expected in expected_files
                )
                
                # 더 관대하게: top_k 내에 있으면 정답
                is_correct = any(
                    any(expected in rf for expected in expected_files)
                    for rf in retrieved_files
                )
                
                if is_correct:
                    correct += 1
                
                details.append({
                    "question": question,
                    "expected": expected_files,
                    "retrieved": retrieved_files,
                    "correct": is_correct,
                })
                
                status = "✅" if is_correct else "❌"
                print(f"  [{i}/{total}] {status} {question[:40]}")
                
            except Exception as e:
                print(f"  [{i}/{total}] ⚠️ 오류: {question[:40]} - {e}")
                details.append({
                    "question": question,
                    "expected": expected_files,
                    "retrieved": [],
                    "correct": False,
                    "error": str(e),
                })
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n  📈 정확도: {accuracy:.2%} ({correct}/{total})")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": details,
        }
    
    def run_full_evaluation(self, chunk_configs: List[Dict[str, int]]):
        """
        여러 청킹 설정에 대해 전체 평가 수행 (learning_ai 컬렉션 재사용)
        
        Args:
            chunk_configs: [
                {"md_chunk_size": 800, "md_chunk_overlap": 100, ...},
                ...
            ]
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []
        
        print("\n" + "="*70)
        print("🚀 청킹 전략 자동 평가 시작")
        print("="*70)
        print(f"테스트 설정 수: {len(chunk_configs)}")
        print(f"평가 질문 수: {len(EVALUATION_DATASET)}")
        print(f"컬렉션: {self.collection_name} (재사용)")
        
        for idx, config in enumerate(chunk_configs, 1):
            print(f"\n\n{'#'*70}")
            print(f"# 설정 {idx}/{len(chunk_configs)}")
            print(f"{'#'*70}")
            
            try:
                # 1. Ingestion (learning_ai 컬렉션 재생성)
                self.run_ingestion(config)
                
                # 2. Evaluation
                eval_result = self.evaluate_collection(
                    EVALUATION_DATASET,
                    top_k=5
                )
                
                # 3. 결과 저장
                results.append({
                    "config": config,
                    "collection_name": self.collection_name,
                    "accuracy": eval_result["accuracy"],
                    "correct": eval_result["correct"],
                    "total": eval_result["total"],
                })
                
            except Exception as e:
                print(f"  ❌ 평가 실패: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "config": config,
                    "error": str(e),
                })
        
        # 결과 저장
        self._save_results(results, timestamp)
        
        # 요약 출력
        self._print_summary(results)
    
    def _save_results(self, results: List[Dict], timestamp: str):
        """결과를 JSON과 CSV로 저장"""
        # JSON
        json_path = self.results_dir / f"eval_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장: {json_path}")
        
        # CSV (간단 버전)
        import csv
        csv_path = self.results_dir / f"eval_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if results and "accuracy" in results[0]:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=["collection_name", "md_chunk_size", "md_chunk_overlap", 
                                "code_chunk_size", "code_chunk_overlap", "accuracy", "correct", "total"]
                )
                writer.writeheader()
                for r in results:
                    if "accuracy" in r:
                        row = {
                            "collection_name": r["collection_name"],
                            **r["config"],
                            "accuracy": r["accuracy"],
                            "correct": r["correct"],
                            "total": r["total"],
                        }
                        writer.writerow(row)
        print(f"💾 CSV 저장: {csv_path}")
    
    def _print_summary(self, results: List[Dict]):
        """결과 요약 출력"""
        print("\n\n" + "="*70)
        print("📊 최종 결과 요약")
        print("="*70)
        
        # 정확도 순으로 정렬
        valid_results = [r for r in results if "accuracy" in r]
        valid_results.sort(key=lambda x: x["accuracy"], reverse=True)
        
        print(f"\n{'순위':<5} {'정확도':<10} {'Markdown Chunk':<20} {'Code Chunk':<20}")
        print("-"*70)
        
        for i, r in enumerate(valid_results, 1):
            cfg = r["config"]
            acc = r["accuracy"]
            md_info = f"{cfg['md_chunk_size']}/{cfg['md_chunk_overlap']}"
            code_info = f"{cfg['code_chunk_size']}/{cfg['code_chunk_overlap']}"
            
            print(f"{i:<5} {acc:>6.1%}     {md_info:<20} {code_info:<20}")
        
        if valid_results:
            best = valid_results[0]
            print("\n🏆 최고 성능 설정:")
            print(f"  - 정확도: {best['accuracy']:.2%}")
            print(f"  - Markdown: size={best['config']['md_chunk_size']}, overlap={best['config']['md_chunk_overlap']}")
            print(f"  - Code: size={best['config']['code_chunk_size']}, overlap={best['config']['code_chunk_overlap']}")


def main():
    """메인 실행 함수"""
    
    # 테스트할 청킹 설정 목록
    # (너무 많으면 시간이 오래 걸리므로 주요 조합만)
    chunk_configs = [
        # 기본 (현재 설정)
        {"md_chunk_size": 1000, "md_chunk_overlap": 100, "code_chunk_size": 800, "code_chunk_overlap": 50},
        
        # Markdown chunk를 작게
        {"md_chunk_size": 600, "md_chunk_overlap": 100, "code_chunk_size": 800, "code_chunk_overlap": 50},
        {"md_chunk_size": 500, "md_chunk_overlap": 100, "code_chunk_size": 800, "code_chunk_overlap": 50},
        
        # Markdown overlap를 크게
        {"md_chunk_size": 800, "md_chunk_overlap": 150, "code_chunk_size": 800, "code_chunk_overlap": 50},
        
        # 둘 다 최적화
        {"md_chunk_size": 600, "md_chunk_overlap": 150, "code_chunk_size": 600, "code_chunk_overlap": 100},
    ]
    
    evaluator = ChunkingEvaluator()
    evaluator.run_full_evaluation(chunk_configs)


if __name__ == "__main__":
    main()
