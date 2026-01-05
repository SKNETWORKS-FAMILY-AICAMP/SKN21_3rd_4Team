"""
기존 Qdrant 컬렉션 성능 평가 스크립트

사용법:
    python scripts/evaluate_existing.py
    
기능:
- 기존 learning_ai 컬렉션의 검색 정확도 평가
- 청킹 설정 변경 없이 현재 상태 평가
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.eval_dataset import EVALUATION_DATASET
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings


class ExistingCollectionEvaluator:
    """기존 컬렉션 평가 클래스"""
    
    def __init__(
        self, 
        collection_name: str = "learning_ai",
        host: str = "localhost",
        port: int = 6333,
    ):
        # .env 파일 로드
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.results_dir = project_root / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def evaluate(self, test_questions: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        """
        컬렉션의 검색 정확도 측정
        
        Returns:
            {
                "accuracy": 0.85,
                "correct": 18,
                "total": 21,
                "details": [...]
            }
        """
        print(f"\n{'='*70}")
        print(f"📊 {self.collection_name} 컬렉션 평가 시작")
        print(f"{'='*70}")
        print(f"평가 질문 수: {len(test_questions)}")
        print(f"Top-K: {top_k}")
        print()
        
        correct = 0
        total = len(test_questions)
        details = []
        
        for i, qa in enumerate(test_questions, 1):
            question = qa["question"]
            expected_files = qa["expected_files"]
            topic = qa.get("topic", "")
            
            try:
                # 질문을 벡터로 변환
                query_vector = self.embedding.embed_query(question)
                
                # Qdrant 검색
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                )
                
                # 검색된 파일명 추출
                retrieved_files = []
                for point in search_results.points:
                    # 디버깅: payload 구조 출력 (첫 번째 질문만)
                    if i == 1 and len(retrieved_files) == 0:
                        print(f"\n  🔍 [디버깅] Payload 구조:")
                        print(f"      {list(point.payload.keys())}")
                        print(f"      전체: {point.payload}")
                    
                    # 메타데이터 구조에 따라 파일명 추출
                    # 강의자료: metadata.source_file
                    # Python docs: metadata.title
                    metadata = point.payload.get("metadata", {})
                    source_file = (
                        point.payload.get("source_file") or
                        metadata.get("source_file") or
                        metadata.get("title") or
                        ""
                    )
                    if source_file:
                        retrieved_files.append(source_file)
                
                # 정답 확인 (top_k 내에 기대 파일이 있으면 정답)
                is_correct = any(
                    any(expected in rf for expected in expected_files)
                    for rf in retrieved_files
                )
                
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  [{i:2d}/{total}] {status} [{topic:12s}] {question[:45]}")
                if not is_correct:
                    print(f"         기대: {expected_files[0][:60]}")
                    if retrieved_files:
                        print(f"         검색: {retrieved_files[0][:60]}")
                    else:
                        print(f"         검색: (결과 없음)")
                
                details.append({
                    "question": question,
                    "topic": topic,
                    "expected": expected_files,
                    "retrieved": retrieved_files,
                    "correct": is_correct,
                })
                
            except Exception as e:
                print(f"  [{i:2d}/{total}] ⚠️ 오류: {question[:40]} - {e}")
                details.append({
                    "question": question,
                    "topic": topic,
                    "expected": expected_files,
                    "retrieved": [],
                    "correct": False,
                    "error": str(e),
                })
        
        accuracy = correct / total if total > 0 else 0
        
        print()
        print("="*70)
        print(f"📈 최종 결과")
        print("="*70)
        print(f"정확도: {accuracy:.1%} ({correct}/{total})")
        
        # 주제별 정확도
        topic_stats = {}
        for d in details:
            topic = d.get("topic", "기타")
            if topic not in topic_stats:
                topic_stats[topic] = {"correct": 0, "total": 0}
            topic_stats[topic]["total"] += 1
            if d["correct"]:
                topic_stats[topic]["correct"] += 1
        
        print("\n주제별 정확도:")
        for topic, stats in sorted(topic_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {topic:15s}: {acc:5.1%} ({stats['correct']}/{stats['total']})")
        
        return {
            "collection_name": self.collection_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "topic_stats": topic_stats,
            "details": details,
        }
    
    def save_results(self, results: Dict[str, Any]):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 저장
        json_path = self.results_dir / f"eval_{self.collection_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장: {json_path}")
        
        # 간단한 텍스트 리포트
        report_path = self.results_dir / f"eval_{self.collection_name}_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"컬렉션: {results['collection_name']}\n")
            f.write(f"정확도: {results['accuracy']:.1%}\n")
            f.write(f"정답/전체: {results['correct']}/{results['total']}\n\n")
            
            f.write("주제별 정확도:\n")
            for topic, stats in results.get("topic_stats", {}).items():
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                f.write(f"  {topic}: {acc:.1%} ({stats['correct']}/{stats['total']})\n")
            
            f.write("\n상세 결과:\n")
            for i, d in enumerate(results["details"], 1):
                status = "✅" if d["correct"] else "❌"
                f.write(f"[{i}] {status} {d['question']}\n")
                if not d["correct"]:
                    f.write(f"    기대: {d['expected']}\n")
                    f.write(f"    검색: {d['retrieved'][:2] if d['retrieved'] else []}\n")
        
        print(f"💾 리포트 저장: {report_path}")


def main():
    """메인 실행"""
    evaluator = ExistingCollectionEvaluator(collection_name="learning_ai")
    
    # 평가 실행
    results = evaluator.evaluate(EVALUATION_DATASET, top_k=5)
    
    # 결과 저장
    evaluator.save_results(results)
    
    print("\n" + "="*70)
    print("✅ 평가 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
