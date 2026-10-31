from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    openai_api_key: str
    telegram_bot_token: str
    model: str = "gpt-6.1-sol"
    # Used only for reasoning models (gpt-5+, gpt-6+, o-series); "low" keeps replies fast.
    reasoning_effort: str = "low"
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
    crisis_phone: str = "112"

    # Paths are absolute so the bot finds telegram_bot/.env regardless of the
    # working directory; the later file wins, the root .env is a fallback.
    model_config = SettingsConfigDict(
        env_file=(BOT_DIR.parent / ".env", BOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
