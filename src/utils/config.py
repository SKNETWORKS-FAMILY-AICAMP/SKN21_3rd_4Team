
###
# from config import Config
# print(Config.OPENAI_API_KEY)
###

import os
from dotenv import load_dotenv
from huggingface_hub import login
from qdrant_client.models import Distance

# .env 파일 로드 (파일이 없으면 무시됨)
load_dotenv()
login(os.getenv('HUGGINGFACE_API_KEY'))

class ConfigAPI:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class ConfigDB:
    HOST = "localhost"
    PORT = "6333"
    COLLECTION_NAME = "learning_ai"
    EMBEDDING_MODEL = "text-embedding-3-large"
    
    # 모델별 벡터 크기 자동 매핑
    VECTOR_SIZE_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    VECTOR_SIZE = VECTOR_SIZE_MAP.get(EMBEDDING_MODEL)
    DISTANCE_FUNCTION = Distance.COSINE

    SNAPSHOT_FOLDER_PATH = "./data/snapshots/"


class ConfigLLM:
    OPENAI_MODEL = "gpt-4o-mini"
