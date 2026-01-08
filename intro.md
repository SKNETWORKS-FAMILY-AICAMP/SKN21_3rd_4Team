## 개발환경 셋팅
```bash
uv venv
source .venv/bin/activate
```
- 패키지 설치
```bash
uv pip install -U sentence-transformers transformers

uv pip install -r requirements.txt
```

### .env 생성 > api key 셋팅
```
# DO NOT STORE SECRETS IN REPO. Put keys in a local `.env` file (not committed).
# Example `.env` (local only - add `.env` to .gitignore if not already present):
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY_HERE>
HUGGINGFACE_API_KEY=<YOUR_HUGGINGFACE_API_KEY_HERE>

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY='<langsmith-api-key>'
LANGSMITH_PROJECT=SKN21_3rd_4Team

TAVILY_API_KEY='<tavily-api-key>'

PYTHONPATH=./
```

## DB 셋팅
### Docker로 QDrant 서버 실행
```bash
docker login
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Qdrant 셋팅
```bash
python init_setting.py
```

## 실행
```bash
python app.py
```