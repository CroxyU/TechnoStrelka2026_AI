import os
from core.text_processor import process_book
from core.vector_store import add_chunks, reset_collection

BOOKS_DIR = "books"

def index_all_books(clear_existing=False):
    """
    Индексирует все книги из папки BOOKS_DIR.
    Если clear_existing=True, сначала очищает коллекцию.
    """
    if not os.path.exists(BOOKS_DIR):
        print(f"Папка {BOOKS_DIR} не найдена.")
        return

    # Получаем список .txt файлов
    txt_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith('.txt')]
    if not txt_files:
        print("Нет книг для индексации.")
        return

    if clear_existing:
        reset_collection()

    total_chunks = 0
    for filename in txt_files:
        filepath = os.path.join(BOOKS_DIR, filename)
        book_id = os.path.splitext(filename)[0]  # имя файла без расширения
        print(f"\nОбрабатываем: {filename}")
        chunks = process_book(filepath, book_id)
        if chunks:
            add_chunks(chunks)
            total_chunks += len(chunks)
        else:
            print(f"Не удалось получить чанки из {filename}")

    print(f"\nИндексация завершена. Всего добавлено чанков: {total_chunks}")

if __name__ == "__main__":
    # При первом запуске можно очистить базу (clear_existing=True)
    # При последующих, если не хотите терять старые книги, ставьте False
    index_all_books(clear_existing=True)