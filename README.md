# Book RAG Bot

Telegram-бот для умного поиска по книгам с использованием RAG (Retrieval-Augmented Generation).

## Возможности (в разработке)
- Загрузка книг в формате `.txt`.
- Поиск фрагментов по запросу.
- Ответы на вопросы по содержанию книг с указанием источников.

## Технологии
- Python 3.10+
- python-telegram-bot (asyncio)
- sentence-transformers
- ChromaDB
- (будет добавлено)

## Запуск
1. Клонировать репозиторий.
2. Установить зависимости: `pip install -r requirements.txt`.
3. Создать файл `.env` с `BOT_TOKEN`.
4. Запустить: `python -m bot.main`.