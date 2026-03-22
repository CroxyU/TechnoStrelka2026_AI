import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, ContextTypes, CallbackQueryHandler
from bot.config import BOT_TOKEN # type: ignore
from core.vector_store import search_with_parents, expanded_search # type: ignore
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
 
    app.add_handler(CommandHandler("ask", ask_command))

    app.add_handler(CallbackQueryHandler(reset_confirmation_callback, pattern="^(confirm_reset|cancel_reset)$"))
    app.add_handler(CallbackQueryHandler(clear_confirmation_callback, pattern="^(confirm_clear|cancel_clear)$"))

    print("Бот запущен...")
    app.run_polling()



if __name__ == "__main__":
    main()