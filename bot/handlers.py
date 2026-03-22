import os
import asyncio
from typing import List, Dict
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes, ConversationHandler
from core.vector_store import ( # type: ignore
    hybrid_search, get_all_books, delete_book, add_chunks,
    add_book_with_parents, clear_database, load_standard_books,
    expanded_search, search_with_parents, get_chroma_client
)  
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
    query = ' '.join(context.args)
    if not query:
        await update.message.reply_text("Укажите запрос после /search")
        return
    
    loop = asyncio.get_event_loop()
    books = await loop.run_in_executor(None, get_all_books)
    if not books:
        await update.message.reply_text("📭 Нет загруженных книг")
        return
    
    await update.message.chat.send_action(action="typing")
    
    # Используем поиск с родительскими документами
    results = await loop.run_in_executor(None, search_with_parents, query, 5)
    
    if not results:
        await update.message.reply_text("По вашему запросу ничего не найдено.")
        return
    
    # Форматируем результат (уже без ограничения на длину текста)
    response = format_search_results(results)
    await update.message.reply_text(response, parse_mode="HTML")


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

    results = await loop.run_in_executor(None, expanded_search, question, 15)

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




def format_search_results(results: List[Dict]) -> str:
    if not results:
        return "Ничего не найдено."
    lines = [f"🔍 <b>Найдено {len(results)} фрагментов:</b>\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"<b>{i}. Книга:</b> {r['book_id']}\n"
            f"<b>Глава:</b> {r.get('chapter', '—')}\n"
            f"<b>Фрагмент:</b>\n<blockquote>{r['text'][:500]}...</blockquote>\n"
            f"<b>Релевантность:</b> {r['score']:.3f}\n"
        )
    return "\n".join(lines)

def format_answer_with_sources(answer: str, results: List[Dict]) -> str:
    """Форматирует ответ LLM вместе с исходными цитатами."""
    lines = [f"🤖 <b>Ответ:</b>\n{answer}\n"]
    lines.append("\n📚 <b>Источники:</b>")
    for i, r in enumerate(results, 1):
        lines.append(
            f"<b>{i}. Книга:</b> {r['book_id']}\n"
            f"<b>Глава:</b> {r.get('chapter', '—')}\n"
            f"<b>Фрагмент:</b>\n<blockquote>{r['text'][:500]}...</blockquote>\n"
            f"<b>Релевантность:</b> {r['score']:.3f}\n"
        )
    return "\n".join(lines)




# --- Загрузка книги ---

async def addbook_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начало загрузки книги. Переводим в состояние ожидания файла."""
    await update.message.reply_text("Отправьте мне файл с книгой в формате .txt")
    return WAITING_FOR_BOOK

async def addbook_receive_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получение файла с иерархической индексацией."""
    if not update.message.document:
        await update.message.reply_text("Это не файл.")
        return WAITING_FOR_BOOK
    
    file = update.message.document
    if not file.file_name.endswith('.txt'):
        await update.message.reply_text("Пожалуйста, отправьте файл .txt")
        return WAITING_FOR_BOOK
    
    await update.message.reply_text("Файл получен. Начинаю обработку...")
    
    file_path = os.path.join("books", file.file_name)
    os.makedirs("books", exist_ok=True)
    await file.get_file().download_to_drive(file_path)
    
    loop = asyncio.get_event_loop()
    try:
        book_id = os.path.splitext(file.file_name)[0]
        
        # Удаляем старую версию, если есть
        await loop.run_in_executor(None, delete_book, book_id)
        
        # Добавляем новую с иерархией
        await update.message.reply_text("Индексирую книгу (создаю иерархию)...")
        await loop.run_in_executor(None, add_book_with_parents, file_path, book_id)
        
        await update.message.reply_text(f"✅ Книга «{book_id}» успешно загружена и проиндексирована!")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        pass
    
    return ConversationHandler.END


async def addbook_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена загрузки"""
    await update.message.reply_text("Загрузка отменена.")
    return ConversationHandler.END




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
    """Сброс к стандартным книгам."""
    keyboard = [
        [InlineKeyboardButton("✅ Да, сбросить", callback_data="confirm_reset")],
        [InlineKeyboardButton("❌ Отмена", callback_data="cancel_reset")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "⚠️ Вы уверены, что хотите сбросить базу?\n"
        "Все текущие книги будут удалены, загрузятся стандартные.",
        reply_markup=reply_markup
    )

async def reset_confirmation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "cancel_reset":
        await query.edit_message_text("❌ Сброс отменён.")
        return
    
    await query.edit_message_text("🔄 Начинаю сброс...")
    
    loop = asyncio.get_event_loop()
    
    await query.message.reply_text("🧹 Очищаю базу...")
    await loop.run_in_executor(None, clear_database)
    
    # Также очищаем родительскую коллекцию
    await loop.run_in_executor(None, clear_parent_collection)
    
    await query.message.reply_text("📚 Загружаю стандартные книги...")
    await loop.run_in_executor(None, load_standard_books)
    
    await query.message.reply_text("✅ Сброс завершён!")




def clear_parent_collection():
    """Очищает коллекцию родительских документов."""
    try:
        client = get_chroma_client()
        client.delete_collection("parent_documents")
        global _parent_collection
        _parent_collection = None
        print("Родительская коллекция удалена.")
    except Exception as e:
        print(f"Ошибка при удалении родительской коллекции: {e}")