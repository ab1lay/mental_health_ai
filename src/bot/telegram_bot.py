from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from src.core.graph import run_graph


class TelegramBot:
    def __init__(self, token: str):
        self.app = ApplicationBuilder().token(token).build()
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Привет! 👋 Напиши, как ты себя чувствуешь.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = update.message.text
        result = await run_graph(text)
        await update.message.reply_text(result)

    def run(self) -> None:
        """Синхронный запуск polling — без конфликтов event loop"""
        self.app.run_polling()
