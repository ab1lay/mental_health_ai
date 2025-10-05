"""
main.py — точка входа проекта Mental Health AI
"""

import logging
from src.bot.telegram_bot import TelegramBot
from src.utils.logger import setup_logging
from config.settings import settings


def main() -> None:
    """Синхронная точка входа."""
    setup_logging(level=settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info("🚀 Mental Health AI baseline starting...")

    try:
        bot = TelegramBot(token=settings.telegram_bot_token)
        bot.run()  # теперь без await
    except Exception as e:
        logger.exception("❌ Ошибка при запуске бота: %s", e)

    logger.info("✅ Mental Health AI stopped successfully.")


if __name__ == "__main__":
    main()
