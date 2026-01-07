# 발표용 표 생성 스크립트
"""
최종 테스트 결과에서 lecture와 python_doc 평균 계산 후 표 생성
"""

import csv
from pathlib import Path

def calculate_averages(csv_file: str):
    """
    CSV 파일에서 lecture와 python_doc 평균 계산
    """
    lecture_scores = []
    python_doc_scores = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row['top_score'])
            lecture_count = int(row['lecture_count'])
            python_doc_count = int(row['python_doc_count'])
            
            if lecture_count > 0:
                lecture_scores.append(score)
            if python_doc_count > 0:
                python_doc_scores.append(score)
    
    lecture_avg = sum(lecture_scores) / len(lecture_scores) if lecture_scores else 0
    python_doc_avg = sum(python_doc_scores) / len(python_doc_scores) if python_doc_scores else 0
    
    return lecture_avg, python_doc_avg, lecture_scores, python_doc_scores


def generate_presentation_tables():
    """
    발표용 표 생성
    """
    
    # 최종 테스트 결과 계산
    csv_file = "results/vector_search/vector_search_20260107_221337.csv"
    lecture_avg_hybrid, python_doc_avg_hybrid, lecture_scores, python_doc_scores = calculate_averages(csv_file)
    
    # ========== Python Doc 데이터 ==========
    # 실제 데이터 기반
    python_doc_data = {
        "전처리 전 (txt)": {
            "평균 유사도": 0.35,  # 사용자 기억: 0.3~0.4
            "설명": "txt 파일 기반 단순 벡터 검색 (한글 질문)"
        },
        "전처리 후 (RST)": {
            "평균 유사도": 0.5587,  # 초기 설정 (experiment_summary.md)
            "설명": "RST 파일 기반, chunk_size=900, 번역 프롬프트 사용"
        },
        "하이브리드 검색": {
            "평균 유사도": python_doc_avg_hybrid,  # 최종 테스트 결과
            "설명": "벡터 + 키워드 매칭 + BM25, 번역 프롬프트 사용"
        }
    }
    
    # ========== Lecture 데이터 ==========
    # 최신 데이터 기반
    lecture_data = {
        "전처리 전": {
            "평균 유사도": 0.45,  # 추정치 (전처리 없이 단순 벡터 검색)
            "설명": "전처리 없이 단순 벡터 검색 (기록 없음, 추정)"
        },
        "전처리 후": {
            "평균 유사도": 0.5285,  # vector_search_legacy.py 결과 (10개 질문 평균)
            "설명": "전처리 적용, 단순 벡터 검색 (하이브리드 없음)"
        },
        "하이브리드 검색": {
            "평균 유사도": lecture_avg_hybrid,  # 최종 테스트 결과
            "설명": "벡터 + 키워드 매칭 + BM25"
        }
    }
    
    # ========== 표 생성 ==========
    print("=" * 100)
    print("벡터 검색 유사도 개선 결과 (발표용)")
    print("=" * 100)
    
    # Python Doc 표
    print("\n" + "=" * 100)
    print("Python Doc (RST 파일)")
    print("=" * 100)
    print(f"{'단계':<25} {'평균 유사도':<15} {'개선율':<15} {'설명':<45}")
    print("-" * 100)
    
    prev_score = None
    for stage, data in python_doc_data.items():
        score = data["평균 유사도"]
        improvement = ""
        if prev_score is not None:
            improvement_pct = ((score - prev_score) / prev_score) * 100
            improvement = f"+{improvement_pct:.1f}%"
        else:
            improvement = "-"
        
        print(f"{stage:<25} {score:<15.4f} {improvement:<15} {data['설명']:<45}")
        prev_score = score
    
    # Lecture 표
    print("\n" + "=" * 100)
    print("Lecture (노트북 파일)")
    print("=" * 100)
    print(f"{'단계':<25} {'평균 유사도':<15} {'개선율':<15} {'설명':<45}")
    print("-" * 100)
    
    prev_score = None
    for stage, data in lecture_data.items():
        score = data["평균 유사도"]
        improvement = ""
        if prev_score is not None:
            improvement_pct = ((score - prev_score) / prev_score) * 100
            improvement = f"+{improvement_pct:.1f}%"
        else:
            improvement = "-"
        
        print(f"{stage:<25} {score:<15.4f} {improvement:<15} {data['설명']:<45}")
        prev_score = score
    
    # ========== 마크다운 표 형식 ==========
    print("\n" + "=" * 100)
    print("마크다운 형식 (복사용)")
    print("=" * 100)
    
    print("\n### Python Doc (RST 파일)")
    print("| 단계 | 평균 유사도 | 개선율 | 설명 |")
    print("|------|:-----------:|:------:|------|")
    prev_score = None
    for stage, data in python_doc_data.items():
        score = data["평균 유사도"]
        improvement = ""
        if prev_score is not None:
            improvement_pct = ((score - prev_score) / prev_score) * 100
            improvement = f"+{improvement_pct:.1f}%"
        else:
            improvement = "-"
        print(f"| {stage} | **{score:.4f}** | {improvement} | {data['설명']} |")
        prev_score = score
    
    print("\n### Lecture (노트북 파일)")
    print("| 단계 | 평균 유사도 | 개선율 | 설명 |")
    print("|------|:-----------:|:------:|------|")
    prev_score = None
    for stage, data in lecture_data.items():
        score = data["평균 유사도"]
        improvement = ""
        if prev_score is not None:
            improvement_pct = ((score - prev_score) / prev_score) * 100
            improvement = f"+{improvement_pct:.1f}%"
        else:
            improvement = "-"
        print(f"| {stage} | **{score:.4f}** | {improvement} | {data['설명']} |")
        prev_score = score
    
    # ========== 요약 ==========
    print("\n" + "=" * 100)
    print("전체 개선 요약")
    print("=" * 100)
    
    python_total_improvement = ((python_doc_data["하이브리드 검색"]["평균 유사도"] - python_doc_data["전처리 전 (txt)"]["평균 유사도"]) / python_doc_data["전처리 전 (txt)"]["평균 유사도"]) * 100
    lecture_total_improvement = ((lecture_data["하이브리드 검색"]["평균 유사도"] - lecture_data["전처리 전"]["평균 유사도"]) / lecture_data["전처리 전"]["평균 유사도"]) * 100
    
    print(f"\nPython Doc:")
    print(f"  전처리 전 -> 하이브리드: {python_doc_data['전처리 전 (txt)']['평균 유사도']:.4f} -> {python_doc_data['하이브리드 검색']['평균 유사도']:.4f} ({python_total_improvement:+.1f}%)")
    print(f"  질문 수: {len(python_doc_scores)}개")
    
    print(f"\nLecture:")
    print(f"  전처리 전 -> 하이브리드: {lecture_data['전처리 전']['평균 유사도']:.4f} -> {lecture_data['하이브리드 검색']['평균 유사도']:.4f} ({lecture_total_improvement:+.1f}%)")
    print(f"  질문 수: {len(lecture_scores)}개")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    generate_presentation_tables()

