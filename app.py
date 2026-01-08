# Flask Backend for Bootcamp AI Tutor
# 기능: 학습 에이전트, 스트리밍 응답
# ============================================

# [Flask 핵심 모듈 임포트]
from flask import Flask, render_template, request, jsonify, session, Response
import json
import os
import time
import uuid

# [Flask 앱 인스턴스 생성 - image 폴더를 static으로 사용]
app = Flask(__name__, static_folder='image', static_url_path='/image')

# [비밀 키 설정 및 캐시 비활성화]
app.secret_key = 'bootcamp-ai-tutor-secret-key-2024'
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 템플릿 변경 시 자동 리로드

# ============================================
# 설정 (Configuration)
# ============================================

# [사용 가능한 모드(에이전트) 정의]
from src.quiz_service import QuizService

# [추가] 퀴즈 서비스 초기화
quiz_service = QuizService()

# -------------------------------------------------------------------------
# Flask App 설정
# -------------------------------------------------------------------------
# (Flask 앱은 상단에서 이미 생성됨)

MODES = {
    'learning': {'name': '학습할래용', 'icon': '📚', 'system_prompt': '친절한 학습 튜터로서 답변해주세요.'},
    'quiz': {'name': '퀴즈풀래용', 'icon': '🧩', 'system_prompt': 'None'} 
}

# ... (중략) ...

@app.route('/api/quiz', methods=['GET'])
def get_quiz():
    """
    퀴즈 데이터 반환 API
    Query Params:
      - category: 'python' | 'lecture' | 'all' (default: all)
      - count: int (default: 5)
    """
    category = request.args.get('category', 'all')
    try:
        count = int(request.args.get('count', 5))
    except ValueError:
        count = 5
        
    quizzes = quiz_service.get_quizzes(category, count)
    return jsonify({'success': True, 'quizzes': quizzes})


# ============================================
# Agent Functions (Mode-specific logic)
# ============================================

def learning_agent(message, context=None):
    """
    학습용 에이전트 - LangGraph Workflow 연결
    
    [연결 방식]
    main.py의 main() 함수를 호출하여 실제 LLM 응답 반환
    """
    from main import main
    
    try:
        print(">>> learning_agent 호출됨!", flush=True)
        # LangGraph workflow 실행
        response = main(message)
        
        # messages에서 답변 추출 및 포맷팅
        analyst_result = response.get('analyst_results', [])
        if analyst_result:
            # 마지막 메시지(AI 답변) 객체에서 content 추출
            last_msg = analyst_result[-1]
            raw_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            
            # 딕셔너리 문자열 파싱 시도
            try:
                import ast
                result_dict = ast.literal_eval(raw_content)
                
                # 마크다운 형태로 포맷팅
                answer_text = f"""## 📚 요약
{result_dict.get('summary', '')}

## 💻 코드 설명
{result_dict.get('code_explanation', '')}

## 💡 실습 팁
{result_dict.get('practice_tips', '')}

## 📌 한 줄 정리
> {result_dict.get('one_liner', '')}
"""
            except (ValueError, SyntaxError):
                # 파싱 실패 시 원본 텍스트 사용
                answer_text = raw_content
        else:
            answer_text = "답변을 생성할 수 없습니다."
        
        # search_results에서 sources 추출
        search_results = response.get('search_results', [])
        
        # ─────────────────────────────────────────────────────────────
        # 터미널 로그: Qdrant 검색 결과 출력
        # ─────────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print(f"🔍 [Qdrant 검색 결과] 질문: {message}")
        print("="*60)
        for i, r in enumerate(search_results, 1):
            source = r.get('metadata', {}).get('source', 'Unknown')
            lecture = r.get('metadata', {}).get('lecture_title', 'Unknown')
            score = r.get('score', 0)
            content_preview = r.get('content', '')[:150].replace('\n', ' ')
            print(f"\n📄 [{i}] 유사도: {score}")
            print(f"   출처: {source} | 강의: {lecture}")
            print(f"   내용: {content_preview}...")
        print("="*60 + "\n", flush=True)
        # [Best Match] 내부 자료 카드 데이터 구성
        import re
        sources = []
        for r in search_results[:3]:
            if r.get('score', 0) > 0.5:
                raw_title = r.get('metadata', {}).get('lecture_title', r.get('metadata', {}).get('source', '문서'))
                # 사용자 요청: "==[내부자료(origin)]==" 문구 제거
                # (혹시 모를 공백이나 대소문자 차이까지 유연하게 처리를 위해 re 사용)
                clean_title = re.sub(r'==\[내부자료\(origin\)\]==', '', raw_title, flags=re.IGNORECASE).strip()
                
                # 내용(content) 가져오기 (줄바꿈 공백 등으로 정리)
                raw_content = r.get('content', '')
                # 사용자 요청: "=== [내부 자료 (Original)] ===" 문구 제거
                # 정규식으로 해당 패턴 및 앞뒤 공백 제거
                clean_content = re.sub(r'={2,}\s*\[내부\s*자료\s*\(Original\)\]\s*={2,}', '', raw_content, flags=re.IGNORECASE).strip()
                clean_content = clean_content.replace('\n', ' ').strip()
                
                sources.append({
                    'type': r.get('metadata', {}).get('source', 'IPYNB').upper(),
                    'title': clean_title,
                    'score': r.get('score', 0),
                    'content': clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
                })
        
        # 추천 질문 추출
        suggested_questions = response.get('suggested_questions', [])
        
        # 외부 검색 소스 추출 (web_search_node에서 설정됨)
        # 외부 검색 소스 추출 (web_search_node에서 설정됨)
        # 사용자 요청으로 외부 검색 카드는 표시하지 않음
        web_sources = []
        
        return {
            'text': answer_text,
            'sources': sources,
            'web_sources': web_sources,  # 외부 검색 소스 추가
            'suggested_questions': suggested_questions,  # 추천 질문 추가
            'steps': [
                {'step': 1, 'title': 'Router', 'desc': '질문 유형 분석 및 검색 설정 결정'},
                {'step': 2, 'title': 'Search', 'desc': f'Qdrant에서 {len(search_results)}개 문서 검색'},
                {'step': 3, 'title': 'Analyst', 'desc': 'GPT-4o-mini로 답변 생성 완료'}
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()  # 전체 에러 스택 출력
        # 에러 발생 시 폴백
        return {
            'text': f"⚠️ 오류가 발생했습니다: {str(e)}\n\n백엔드 설정을 확인해주세요.",
            'sources': [],
            'steps': [
                {'step': 1, 'title': '오류', 'desc': f'{str(e)[:50]}...'}
            ]
        }

def get_agent_response(mode, message, context=None):
    """모드에 따라 적절한 에이전트 호출"""
    return learning_agent(message, context)

# ============================================
# 라우트 (Routes) - URL 엔드포인트 정의
# ============================================
# 헬스 체크 API
@app.route('/health')
def health_check():
    """서버 및 DB 연결 상태 확인"""
    try:
        # Qdrant DB 연결 확인 (가벼운 연결 시도)
        import socket
        # 6333 포트(Qdrant 기본 포트)가 열려있는지 확인
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1) # 1초 타임아웃
        result = sock.connect_ex(('localhost', 6333))
        sock.close()
        
        if result == 0:
            return jsonify({'status': 'ok', 'message': '정상 연결'})
        else:
            return jsonify({'status': 'error', 'message': 'DB 연결 실패'}), 503
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html', modes=MODES)

@app.route('/chat', methods=['POST'])
def chat():
    """채팅 API - POST /chat"""
    data = request.get_json()
    message = data.get('message', '')
    mode = data.get('mode', 'learning')
    
    # 에이전트에서 응답 생성
    response = get_agent_response(mode, message)
    
    return jsonify({
        'answer': response['text'],
        'sources': response['sources'],
        'steps': response['steps']
    })

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """스트리밍 채팅 API - POST /chat/stream"""
    data = request.get_json()
    message = data.get('message', '')
    mode = data.get('mode', 'learning')
    
    # 에이전트 응답 생성
    response = get_agent_response(mode, message)
    
    def generate():
        # 1단계: 진행 단계 정보 전송
        for step in response['steps']:
            yield f"data: {json.dumps({'type': 'step', 'data': step})}\n\n"
            time.sleep(0.5)
        
        # 2단계: 답변을 글자 하나씩 전송
        for char in response['text']:
            yield f"data: {json.dumps({'type': 'char', 'data': char})}\n\n"
            time.sleep(0.02)
        
        # 3단계: 추천 질문 전송 (먼저 표시)
        suggested = response.get('suggested_questions', [])
        print(f"🔔 [SSE] 추천 질문: {suggested}", flush=True)
        if suggested:
            yield f"data: {json.dumps({'type': 'suggestions', 'data': suggested})}\n\n"
        
        # 4단계: 참고 자료(카드) 전송 (질문 아래에 표시)
        yield f"data: {json.dumps({'type': 'sources', 'data': response['sources']})}\n\n"

        # 4.5단계: 외부 검색 결과 전송
        web_sources = response.get('web_sources', [])
        if web_sources:
             yield f"data: {json.dumps({'type': 'web_sources', 'data': web_sources})}\n\n"
        
        # 5단계: 완료 신호
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/reset', methods=['POST'])
def reset_all():
    """전체 세션 초기화"""
    session.clear()
    return jsonify({'success': True, 'message': 'Session reset'})

@app.route('/modes')
def get_modes():
    """사용 가능한 모드 목록"""
    return jsonify(MODES)

# ============================================
# 앱 실행
# ============================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)
