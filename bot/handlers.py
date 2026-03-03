import os
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler
from core.vector_store import search, add_chunks
from core.text_processor import process_book

# Состояния для ConversationHandler (если нужно для загрузки книги)
WAITING_FOR_BOOK = 1

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "Привет! Я бот для поиска по книгам.\n"
        "Команды:\n"
        "/search <запрос> — найти фрагменты\n"
        "/addbook — загрузить новую книгу (отправьте файл .txt после команды)\n"
        "/help — показать это сообщение"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)  # просто повторяем то же сообщение

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /search. Ожидает текст после команды."""
    # Получаем текст запроса (всё, что после /search)
    query = ' '.join(context.args)
    if not query:
        await update.message.reply_text("Пожалуйста, укажите запрос после /search, например: /search Наташа Ростова")
        return

    # Отправляем сообщение, что ищем (чтобы пользователь не думал, что бот завис)
    await update.message.reply_text(f"Ищу: «{query}»...")

    # Выполняем поиск (синхронный вызов может заблокировать бота, поэтому обернём в asyncio.to_thread)
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, search, query)  # 5 результатов

    if not results:
        await update.message.reply_text("Ничего не найдено.")
        return

    # Формируем ответ
    response = f"Найдено {len(results)} фрагментов:\n\n"
    for i, r in enumerate(results, 1):
        response += f"**{i}. Книга:** {r['book_id']}\n"
        response += f"**Глава:** {r['chapter']}\n"
        response += f"**Фрагмент:**\n{r['text'][:300]}...\n"
        response += f"**Релевантность:** {r['score']:.2f}\n\n"

    # Разбиваем, если слишком длинное сообщение (Telegram ограничение 4096 символов)
    if len(response) > 4000:
        # Отправляем по частям
        parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
        for part in parts:
            await update.message.reply_text(part)
    else:
        await update.message.reply_text(response)

# --- Загрузка книги ---

async def addbook_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начало загрузки книги. Переводим в состояние ожидания файла."""
    await update.message.reply_text("Отправьте мне файл с книгой в формате .txt")
    return WAITING_FOR_BOOK

async def addbook_receive_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получение файла, сохранение и индексация."""
    if not update.message.document:
        await update.message.reply_text("Это не файл. Пожалуйста, отправьте файл .txt")
        return WAITING_FOR_BOOK

    file = update.message.document
    if not file.file_name.endswith('.txt'):
        await update.message.reply_text("Пожалуйста, отправьте файл с расширением .txt")
        return WAITING_FOR_BOOK

    await update.message.reply_text("Файл получен. Начинаю обработку...")

    # Скачиваем файл
    file_path = os.path.join("books", file.file_name)
    os.makedirs("books", exist_ok=True)
    tg_file = await file.get_file()               # <<< ИСПРАВЛЕНО
    await tg_file.download_to_drive(file_path)    # <<< ИСПРАВЛЕНО

    loop = asyncio.get_event_loop()
    try:
        book_id = os.path.splitext(file.file_name)[0]
        chunks = await loop.run_in_executor(None, process_book, file_path, book_id)
        if not chunks:
            await update.message.reply_text("Не удалось обработать книгу (пустой файл или ошибка).")
            return ConversationHandler.END

        await update.message.reply_text(f"Книга разбита на {len(chunks)} фрагментов. Индексирую...")

        # Удаляем старую версию книги, если есть (опционально)
        from core.vector_store import delete_book, add_chunks
        await loop.run_in_executor(None, delete_book, book_id)
        await loop.run_in_executor(None, add_chunks, chunks)

        await update.message.reply_text(f"Книга «{book_id}» успешно загружена и проиндексирована!")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при обработке книги: {e}")
    finally:
        # Если хотите удалить файл после индексации, раскомментируйте:
        # os.remove(file_path)
        pass

    return ConversationHandler.END
async def addbook_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена загрузки (если пользователь передумал)."""
    await update.message.reply_text("Загрузка отменена.")
    return ConversationHandler.END
	