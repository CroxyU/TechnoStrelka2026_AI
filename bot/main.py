import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, ContextTypes
from bot.config import BOT_TOKEN
from bot.handlers import (
    start, help_command, search_command,
    addbook_start, addbook_receive_file, addbook_cancel,
    WAITING_FOR_BOOK
)

def main():
    # Создаём приложение
    app = Application.builder().token(BOT_TOKEN).build()

    # Обработчик команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("search", search_command))

    # ConversationHandler для загрузки книги
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("addbook", addbook_start)],
        states={
            WAITING_FOR_BOOK: [
                MessageHandler(filters.Document.ALL, addbook_receive_file),
                CommandHandler("cancel", addbook_cancel)
            ]
        },
        fallbacks=[CommandHandler("cancel", addbook_cancel)]
    )
    app.add_handler(conv_handler)

    # Обработчик текстовых сообщений (не команд) — можно использовать как быстрый поиск
    async def text_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Если пользователь просто написал текст, воспринимаем это как поисковый запрос
        # Но чтобы не путать с другими сообщениями, можно добавить проверку
        query = update.message.text
        # Отправляем "печатает..."
        await update.message.chat.send_action(action="typing")
        # Выполняем поиск (аналогично search_command)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, search, query, 5)

        if not results:
            await update.message.reply_text("Ничего не найдено.")
            return

        response = f"Найдено {len(results)} фрагментов:\n\n"
        for i, r in enumerate(results, 1):
            response += f"**{i}. Книга:** {r['book_id']}\n"
            response += f"**Глава:** {r['chapter']}\n"
            response += f"**Фрагмент:**\n{r['text'][:300]}...\n"
            response += f"**Релевантность:** {r['score']:.2f}\n\n"

        await update.message.reply_text(response)

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_search))

    # Запускаем бота
    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()