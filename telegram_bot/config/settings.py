from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    telegram_bot_token: str
    model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    use_webhook: bool = False  # переключатель
    webhook_host: str | None = None
    webhook_port: int = 8443
    webhook_path: str | None = None
    proactive_enabled: bool = True
    proactive_check_interval_minutes: int = 30
    proactive_idle_hours: int = 24
    proactive_min_user_messages: int = 2
    proactive_batch_size: int = 20
    patient_portal_base_url: str = "http://127.0.0.1:8000"
    patient_portal_token_ttl_minutes: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
