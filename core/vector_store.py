import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from typing import List, Dict, Optional

# Константы
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Лёгкая и быстрая модель
CHROMA_PERSIST_DIR = "./chroma_data"   # Папка для хранения базы данных
COLLECTION_NAME = "books"               # Имя коллекции
MIN_SCORE = 0.3
# Глобальные объекты (чтобы не перезагружать модель и клиент при каждом вызове)
_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
    """Ленивая загрузка модели эмбеддингов."""
    global _model
    if _model is None:
        print(f"Загружаем модель эмбеддингов: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_collection():
    """Ленивая инициализация клиента ChromaDB и получение коллекции."""
    global _chroma_client, _collection
    if _chroma_client is None:
        # Убеждаемся, что папка для данных существует
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if _collection is None:
        # Получаем или создаём коллекцию
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # используем косинусную метрику
        )
    return _collection


def add_chunks(chunks: List[Dict]):
    """
    Добавляет список чанков в векторную базу.
    Каждый чанк — словарь с ключами: book_id, chapter, chunk_index, text
    """
    if not chunks:
        return

    model = get_embedding_model()
    collection = get_chroma_collection()

    # Подготавливаем данные для пакетной вставки
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    # Собираем тексты для эмбеддингов
    texts = [chunk['text'] for chunk in chunks]

    # Генерируем эмбеддинги пачкой (быстрее, чем по одному)
    print("Генерация эмбеддингов...")
    embeddings_list = model.encode(texts, show_progress_bar=True).tolist()

    for idx, chunk in enumerate(tqdm(chunks, desc="Подготовка данных")):
        # Уникальный ID: book_id + _ + chunk_index
        chunk_id = f"{chunk['book_id']}_{chunk['chunk_index']}"
        ids.append(chunk_id)
        embeddings.append(embeddings_list[idx])
        metadatas.append({
            "book_id": chunk['book_id'],
            "chapter": chunk['chapter'],
            "chunk_index": chunk['chunk_index']
        })
        documents.append(chunk['text'])

    # Добавляем в коллекцию
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    print(f"Добавлено {len(chunks)} чанков в коллекцию.")


def search(query: str, n_results: int = 5) -> List[Dict]:
    """
    Ищет n_results наиболее похожих чанков на запрос.
    Возвращает список словарей с ключами:
        book_id, chapter, chunk_index, text, score
    """
    if not query.strip():
        return []

    model = get_embedding_model()
    collection = get_chroma_collection()

    # Преобразуем запрос в вектор
    query_emb = model.encode(query).tolist()

    # Выполняем поиск
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )

    # Обрабатываем результаты
    output = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            # distance — косинусное расстояние (от 0 до 2). Преобразуем в сходство.
            distance = results['distances'][0][i]
            score = 1 - distance  # чем ближе к 1, тем лучше
            if score > MIN_SCORE:
                output.append({
                    'book_id': results['metadatas'][0][i]['book_id'],
                    'chapter': results['metadatas'][0][i]['chapter'],
                    'chunk_index': results['metadatas'][0][i]['chunk_index'],
                    'text': results['documents'][0][i],
                    'score': score
                })
            else:
                break
    return output

def delete_book(book_id: str):
    """Удаляет все чанки, принадлежащие книге с указанным book_id."""
    collection = get_chroma_collection()
    # Удаляем записи, где метаданные содержат book_id = заданное значение
    collection.delete(where={"book_id": book_id})
    print(f"Все записи для книги '{book_id}' удалены.")

def reset_collection():
    """Удаляет коллекцию (для пересоздания базы)."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        global _collection
        _collection = None
        print(f"Коллекция {COLLECTION_NAME} удалена.")
    except ValueError:
        print("Коллекция не существовала.")