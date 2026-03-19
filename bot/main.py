import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, ContextTypes, CallbackQueryHandler
from bot.config import BOT_TOKEN # type: ignore
from core.vector_store import search, add_chunks # type: ignore
from bot.handlers import ( # type: ignore
    start, help_command, search_command,
    addbook_start, addbook_receive_file, addbook_cancel,
    WAITING_FOR_BOOK, ask_command, listbooks_command, cleardb_command,
    reset_command, reset_confirmation_callback,
    clear_confirmation_callback
)

import logging
import warnings
import os
# Отключаем предупреждения
warnings.filterwarnings("ignore")

# Устанавливаем уровень логирования для библиотек
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_WARNINGS"] = "1"
def main():
    # Создаём приложение
    app = Application.builder().token(BOT_TOKEN).read_timeout(30).write_timeout(30).connect_timeout(30).pool_timeout(30).build()

    # Обработчик команд
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(CommandHandler("listbooks", listbooks_command))
    app.add_handler(CommandHandler("cleardb", cleardb_command))
    app.add_handler(CommandHandler("reset", reset_command))


    
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
    # Обработчик текстовых сообщений
    async def text_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Если пользователь написал текст, воспринимаем это как поисковый запрос
        query = update.message.text
        # Отправляем "печатает..."
        await update.message.chat.send_action(action="typing")
        # Выполняем поиск
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, search, query, 5)

        if not results:
            await update.message.reply_text("Ничего не найдено.")
            return

        response = f"Найдено {len(results)} фрагментов:\n\n"
        for i, r in enumerate(results, 1):
            response += f"**{i}. Книга:** {r['book_id']}\n"
            response += f"**Фрагмент:**\n{r['text'][:300]}...\n"
            response += f"**Релевантность:** {r['score']:.2f}\n\n"

        await update.message.reply_text(response)

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_search))

    app.add_handler(CommandHandler("ask", ask_command))

    app.add_handler(CallbackQueryHandler(reset_confirmation_callback, pattern="^(confirm_reset|cancel_reset)$"))
    app.add_handler(CallbackQueryHandler(clear_confirmation_callback, pattern="^(confirm_clear|cancel_clear)$"))

    print("Бот запущен...")
    app.run_polling()



if __name__ == "__main__":
    main()