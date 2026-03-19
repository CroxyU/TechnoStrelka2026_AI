import os
import asyncio
from typing import List, Dict
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes, ConversationHandler
from core.vector_store import hybrid_search, get_all_books, clear_database, load_standard_books, expanded_search  # type: ignore
from core.text_processor import process_book # type: ignore
from core.llm_client import generate_answer  # type: ignore # в начало файла
WAITING_FOR_BOOK = 1


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    welcome_text = (
            "📚 <b>Добро пожаловать в БЯМ RAG ТЕХНОСТРЕЛКА!</b>\n\n"
    "Я помогу вам искать информацию в книгах и отвечу на вопросы по их содержанию.\n\n"
    "🔍 <b>Что я умею:</b>\n"
    "• Искать фрагменты текста по запросу\n"
    "• Отвечать на вопросы, опираясь на книги\n"
    "• Загружать новые книги (формат .txt)\n"
    "• Показывать список уже загруженных книг\n\n"
    "⬇️ Используйте команды:\n"
    "/search, /ask, /addbook, /listbooks, /help\n"
    "(Это решение кейса на 'Технострелка 2026') "
    )
    await update.message.reply_text(
        welcome_text,
        parse_mode="HTML"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
         "📖 <b>Справка по командам</b>\n\n"
    "<b>Основные команды:</b>\n"
    "• /start – приветствие и главное меню\n"
    "• /help – это сообщение\n"
    "• /search [запрос] – поиск фрагментов (например, /search Наташа Ростова)\n"
    "• /ask [вопрос] – задать вопрос по книгам (например, /ask Кто такой Пьер?)\n"
    "• /addbook – загрузить новую книгу (отправьте .txt файл после команды)\n"
    "• /listbooks – показать список всех загруженных книг\n\n"
    "<b>Административные команды:</b>\n"
    "• /cleardb – полностью очистить базу данных (с подтверждением)\n"
    "• /reset – сбросить базу к стандартному набору книг\n\n"
    "После загрузки книги она автоматически индексируется и становится доступной для поиска и вопросов.\n"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /search. Ожидает текст после команды."""
    # Получаем текст запроса
    query = ' '.join(context.args)
    if not query:
        await update.message.reply_text("Пожалуйста, укажите запрос после /search, например: /search Наташа Ростова")
        return

   # Проверяем наличие книг в базе
    loop = asyncio.get_event_loop()
    books = await loop.run_in_executor(None, get_all_books)
    if not books:
        await update.message.reply_text(
            "📭 В базе пока нет ни одной книги.\n"
            "Загрузите книгу через /addbook и повторите поиск."
        )
        return

    await update.message.chat.send_action(action="typing")
    results = await loop.run_in_executor(None, expanded_search, query, 15)
    
    for r in results:
        print(f"  - {r['book_id']} (score: {r['score']})")
    if not results:
        await update.message.reply_text("По вашему запросу ничего не найдено.")
        return

    response = format_search_results(results)

    # Разбиваем, если слишком длинное сообщение
    if len(response) > 4000:
        # Отправляем по частям
        parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
        for part in parts:
            await update.message.reply_text(part, parse_mode="HTML")
    else:
        await update.message.reply_text(response, parse_mode="HTML")

def format_search_results(results: List[Dict]) -> str:
    """Форматирует список результатов поиска в читаемый вид с HTML."""
    if not results:
        return "Ничего не найдено."

    lines = [f"🔍 <b>Найдено {len(results)} фрагментов:</b>\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"<b>{i}. Книга:</b> {r['book_id']}\n"
            f"<b>Фрагмент:</b>\n<blockquote>{r['text'][:300]}...</blockquote>\n"
            f"<b>Релевантность:</b> {r['score']:.2f}\n"
        )
    return "\n".join(lines)

def format_answer_with_sources(answer: str, results: List[Dict]) -> str:
    """Форматирует ответ LLM вместе с исходными цитатами."""
    lines = [f"🤖 <b>Ответ:</b>\n{answer}\n"]
    lines.append("\n📚 <b>Источники:</b>")
    for i, r in enumerate(results, 1):
        lines.append(
            f"\n{i}. <b>{r['book_id']}</b>"
            f"<blockquote>{r['text'][:200]}...</blockquote>"
        )
    return "\n".join(lines)

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
    tg_file = await file.get_file()               
    await tg_file.download_to_drive(file_path)    

    loop = asyncio.get_event_loop()
    try:
        book_id = os.path.splitext(file.file_name)[0]
        chunks = await loop.run_in_executor(None, process_book, file_path, book_id)
        if not chunks:
            await update.message.reply_text("Не удалось обработать книгу (пустой файл или ошибка).")
            return ConversationHandler.END

        await update.message.reply_text(f"Книга разбита на {len(chunks)} фрагментов. Индексирую...")

        # Удаляем старую версию книги (если есть)
        from core.vector_store import delete_book, add_chunks # type: ignore
        await loop.run_in_executor(None, delete_book, book_id)
        await loop.run_in_executor(None, add_chunks, chunks)

        await update.message.reply_text(f"Книга «{book_id}» успешно загружена и проиндексирована!")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при обработке книги: {e}")
    finally:
        pass

    return ConversationHandler.END

async def addbook_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена загрузки"""
    await update.message.reply_text("Загрузка отменена.")
    return ConversationHandler.END

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = ' '.join(context.args)
    if not question:
        await update.message.reply_text("Пожалуйста, укажите вопрос после /ask")
        return

    await update.message.reply_text(f"Ищу информацию по вопросу: «{question}»...")

    
    # Ищем релевантные чанки
    loop = asyncio.get_event_loop()

      # Проверяем наличие книг
    books = await loop.run_in_executor(None, get_all_books)
    if not books:
        await update.message.reply_text(
            "📭 В базе пока нет ни одной книги.\n"
            "Загрузите книгу через /addbook, чтобы я мог отвечать на вопросы."
        )
        return

    results = await loop.run_in_executor(None, expanded_search, 15)

    if not results:
        await update.message.reply_text("Я не нашёл в книгах информации, которая могла бы помочь ответить на этот вопрос.")
        return

    # Генерируем ответ
    await update.message.reply_text("Нашёл несколько фрагментов. Формирую ответ...")
    try:
        answer = await loop.run_in_executor(None, generate_answer, question, results)
    except Exception as e:
        await update.message.reply_text(f"Ошибка при генерации ответа: {e}")
        return

    # Формируем итоговое сообщение
    answer = await loop.run_in_executor(None, generate_answer, question, results)
    response = format_answer_with_sources(answer, results)
    await update.message.reply_text(response, parse_mode="HTML")

async def listbooks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет пользователю список загруженных книг."""
    await update.message.chat.send_action(action="typing")
    
    # Выполняем получение списка
    loop = asyncio.get_event_loop()
    book_list = await loop.run_in_executor(None, get_all_books)
    
    if not book_list:
        await update.message.reply_text(
            "📚 В базе пока нет ни одной книги.\n"
            "Загрузите книгу через /addbook"
        )
        return
    
    # Формируем ответ
    response = "📚 <b>Загруженные книги:</b>\n\n"
    response += "\n".join(f"• {book}" for book in book_list)
    
    await update.message.reply_text(response, parse_mode="HTML")

async def cleardb_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запрашивает подтверждение очистки базы данных."""
    keyboard = [
        [InlineKeyboardButton("✅ Да, очистить", callback_data="confirm_clear")],
        [InlineKeyboardButton("❌ Отмена", callback_data="cancel_clear")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "⚠️ Вы уверены, что хотите полностью очистить базу данных?\n"
        "Это действие необратимо, все загруженные книги будут удалены.",
        reply_markup=reply_markup
    )

async def clear_confirmation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает подтверждение или отмену очистки базы."""
    query = update.callback_query
    await query.answer()

    if query.data == "cancel_clear":
        await query.edit_message_text("❌ Очистка отменена.")
        return

    if query.data == "confirm_clear":
        await query.edit_message_text("🧹 Начинаю очистку базы данных...")

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, clear_database)
            await query.message.reply_text("✅ База данных полностью очищена.")
        except Exception as e:
            await query.message.reply_text(f"❌ Ошибка при очистке: {e}")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запрашивает подтверждение сброса к стандартным книгам."""
    keyboard = [
        [InlineKeyboardButton("✅ Да, сбросить", callback_data="confirm_reset")],
        [InlineKeyboardButton("❌ Отмена", callback_data="cancel_reset")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "⚠️ Вы уверены, что хотите сбросить базу к стандартным книгам?\n"
        "Все текущие загруженные книги будут удалены, а из папки `standard_books` загрузятся стандартные.",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def reset_confirmation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает подтверждение или отмену сброса."""
    query = update.callback_query
    await query.answer()

    if query.data == "cancel_reset":
        await query.edit_message_text("❌ Сброс отменён.")
        return

    if query.data == "confirm_reset":
        # Сообщаем о начале
        await query.edit_message_text("🔄 Начинаю сброс базы данных к стандартным книгам...")

        loop = asyncio.get_event_loop()

        # Этап 1: Очистка базы
        await query.message.reply_text("🧹 Очищаю текущую базу данных...")
        try:
            await loop.run_in_executor(None, clear_database)
        except Exception as e:
            await query.message.reply_text(f"❌ Ошибка при очистке базы: {e}")
            return

        # Этап 2: Загрузка стандартных книг
        await query.message.reply_text("📚 Загружаю стандартные книги из папки standard_books...")
        try:
            await loop.run_in_executor(None, load_standard_books)
        except Exception as e:
            await query.message.reply_text(f"❌ Ошибка при загрузке книг: {e}")
            return

        # Финальное сообщение
        await query.message.reply_text("✅ Сброс завершён! Теперь доступны стандартные книги.")