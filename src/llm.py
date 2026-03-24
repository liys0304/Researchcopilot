from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
)

def get_chat_model():
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.2,
    )

def get_embedding_model():
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        check_embedding_ctx_length=False,  # 关键
    )