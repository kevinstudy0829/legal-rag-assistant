"""
集中管理所有可調參數。所有其他模組一律從這裡讀取設定，
不要在程式碼各處散落 os.environ.get(...)。
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- 資料庫 ---
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/legal_rag"

    # --- Gemini ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-latest"

    # --- 向量化模型 ---
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_DIM: int = 1024

    # --- 重排序 ---
    ENABLE_RERANK: bool = True
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str = "cpu"

    # --- 混合檢索 ---
    ENABLE_HYBRID_SEARCH: bool = True

    # --- 檢索參數 ---
    TOP_K_CANDIDATES: int = 20
    TOP_K_FINAL: int = 5

    # --- 分塊參數 ---
    CHUNK_MAX_CHARS: int = 800
    CHUNK_OVERLAP_CHARS: int = 150

    # --- 資料來源 ---
    LAW_NAME: str = "中華民國刑法"
    LAW_BULK_XML_URL: str = "https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?DType=RAW_XML&AuData=CF"
    LAW_SOURCE_PAGE_URL: str = "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=C0000001"


settings = Settings()
