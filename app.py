# Flask Backend for Bootcamp AI Tutor
# 기능: 학습 에이전트, 스트리밍 응답
# ============================================

# [Flask 핵심 모듈 임포트]
from flask import Flask, render_template, request, jsonify, session, Response
import json
import os
import time
import uuid

# [Flask 앱 인스턴스 생성]
app = Flask(__name__)

# [비밀 키 설정]
app.secret_key = 'bootcamp-ai-tutor-secret-key-2024'

# ============================================
# 설정 (Configuration)
# ============================================

# [사용 가능한 모드(에이전트) 정의]
MODES = {
    'learning': {'name': '학습할래용', 'icon': '📚', 'system_prompt': '친절한 학습 튜터로서 답변해주세요.'},
}


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
        # LangGraph workflow 실행
        response = main(message)
        
        # analyst_results에서 답변 추출
        analyst_results = response.get('analyst_results', [])
        if analyst_results:
            # HumanMessage 객체에서 content 추출
            answer_text = analyst_results[0].content if hasattr(analyst_results[0], 'content') else str(analyst_results[0])
        else:
            answer_text = "답변을 생성할 수 없습니다."
        
        # search_results에서 sources 추출
        search_results = response.get('search_results', [])
        sources = [
            {
                'type': 'IPYNB',
                'title': r.get('metadata', {}).get('lecture_title', 'Unknown'),
                'content': r.get('content', '')[:100] + '...'
            }
            for r in search_results[:3]  # 상위 3개만
        ]
        
        return {
            'text': answer_text,
            'sources': sources,
            'steps': [
                {'step': 1, 'title': 'Router', 'desc': '질문 유형 분석 및 검색 설정 결정'},
                {'step': 2, 'title': 'Search', 'desc': f'Qdrant에서 {len(search_results)}개 문서 검색'},
                {'step': 3, 'title': 'Analyst', 'desc': 'GPT-4o-mini로 답변 생성 완료'}
            ]
        }
    except Exception as e:
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
        
        # 3단계: 참고 자료 전송
        yield f"data: {json.dumps({'type': 'sources', 'data': response['sources']})}\n\n"
        
        # 4단계: 완료 신호
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
