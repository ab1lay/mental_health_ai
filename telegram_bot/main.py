"""Точка входа для Mental Health AI бота."""

import logging

from config.settings import settings
from src.bot.telegram_bot import TelegramBot
from src.utils.logger import setup_logging


def main() -> None:
    """Запускает Telegram-бота с базовой поддержкой психического здоровья."""
    setup_logging(level=settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Mental Health AI запускается...")

    try:
        bot = TelegramBot(token=settings.telegram_bot_token)
        bot.run()
    except Exception as e:
        logger.exception("Ошибка при запуске бота: %s", e)

    logger.info("Mental Health AI остановлен.")


if __name__ == "__main__":
    main()
