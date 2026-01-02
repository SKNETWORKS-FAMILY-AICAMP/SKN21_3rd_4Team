
###
# from config import Config
# print(Config.OPENAI_API_KEY)
###

import os
from dotenv import load_dotenv
from huggingface_hub import login

# .env 파일 로드 (파일이 없으면 무시됨)
load_dotenv()
login(os.getenv('HUGGINGFACE_API_KEY'))

class ConfigAPI:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class ConfigDB:
    COLLECTION_NAME = "learning_ai"