"""
Configuration settings for the Athena application.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# 获取项目根目录的绝对路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# 加载.env文件
dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # API settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Athena"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./athena.db"
    
    # Vector Database
    VECTOR_DB_PATH: str = "./vector_db"
    
    # LLM API settings
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Security
    SECRET_KEY: str = "changethisinproduction"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = os.path.join(ROOT_DIR, ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()