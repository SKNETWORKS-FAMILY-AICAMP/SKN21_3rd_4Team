## 개발환경 셋팅
- 패키지 설치
```bash
uv pip install -r requirements.txt
```

### langchain > Tavily Search
https://docs.langchain.com/oss/python/integrations/tools
https://docs.langchain.com/oss/python/integrations/tools/tavily_search

### .env > api key 셋팅
```
OPENAI_API_KEY='<your-openai-api-key>'

HUGGINGFACE_API_KEY='<your-huggingface-api-key>'

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY='<langsmith-api-key>'
LANGSMITH_PROJECT=pr-linear-cluster-64

TAVILY_API_KEY='<tavily-api-key>'
```

## DB 셋팅
### Docker로 QDrant 서버 실행
```bash
docker pull skn21-3rd/4team
docker run -p 6333:6333 -p 6334:6334 luccia/skn21-3rd-4team
docker run -p 6333:6333 -p 6334:6334 -v "qdrant_storage:/qdrant/storage:z" skn21-3rd/4team
```