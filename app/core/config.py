from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-5"
    max_tokens: int = 4096

    class Config:
        env_file = ".env"

settings = Settings()