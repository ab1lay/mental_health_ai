from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    telegram_bot_token: str
    model: str = "gpt-4o-mini"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
