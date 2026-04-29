from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-5"
    max_tokens: int = 4096
    brave_search_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()