import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from bot.config import BOT_TOKEN

async def start(update Update, context ContextTypes.DEFAULT_TYPE)
    await update.message.reply_text(Привет! Я бот для поиска по книгам. Пока я ничего не умею, но скоро научусь!)

async def echo(update Update, context ContextTypes.DEFAULT_TYPE)
    await update.message.reply_text(fВы сказали {update.message.text})

def main()
    # Создаём приложение
    app = Application.builder().token(BOT_TOKEN).build()

    # Добавляем обработчики
    app.add_handler(CommandHandler(start, start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Запускаем бота (polling)
    print(Бот запущен...)
    app.run_polling()

if __name__ == __main__
    main()