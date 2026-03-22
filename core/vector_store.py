import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from core.text_processor import process_book_with_parents, process_book # type: ignore
from core.llm_client import expand_query # type: ignore
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Глобальные переменные для BM25
_bm25_index = None
_bm25_chunks = []   # список словарей с полными данными 
# Константы
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Лёгкая и быстрая модель
CHROMA_PERSIST_DIR = "./chroma_data"   # Папка для хранения базы данных
COLLECTION_NAME = "books"               # Имя коллекции
MIN_SCORE = 0.33
# Глобальные объекты
_model = None
_chroma_client = None
_collection = None
_parent_collection = None

def build_bm25_index(chunks: List[Dict]):
    """
    Строит BM25 индекс по списку чанков.
    chunks - список словарей с ключами book_id, chunk_index, text
    """
    global _bm25_index, _bm25_chunks
    _bm25_chunks = chunks
    # Токенизируем текст каждого чанка
    tokenized_corpus = [word_tokenize(chunk['text'], language='russian') for chunk in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    print(f"BM25 индекс построен. Всего чанков: {len(chunks)}")

def get_embedding_model():
    """Ленивая загрузка модели эмбеддингов."""
    global _model
    if _model is None:
        print(f"Загружаем модель эмбеддингов: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_chroma_client():
    """Возвращает клиент ChromaDB."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client

def get_chroma_collection():
    """Ленивая инициализация клиента ChromaDB и получение коллекции."""
    global _chroma_client, _collection
    if _chroma_client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # используем косинусную метрику
        )
    return _collection

def get_parent_collection():
    """Получает или создаёт коллекцию для родительских документов."""
    global _parent_collection
    if _parent_collection is None:
        client = get_chroma_client()
        _parent_collection = client.get_or_create_collection(
            name="parent_documents",
            metadata={"hnsw:space": "cosine"}
        )
    return _parent_collection

def add_book_with_parents(filepath: str, book_id: str):
    """
    Добавляет книгу в базу с иерархической структурой.
    """
    child_chunks, parent_docs = process_book_with_parents(filepath, book_id)
    
    # 1. Добавляем дочерние чанки (для поиска)
    add_child_chunks(child_chunks)
    
    # 2. Добавляем родительские документы (для возврата)
    add_parent_documents(parent_docs)
    
    print(f"Добавлена книга {book_id}: {len(child_chunks)} чанков, {len(parent_docs)} родителей")

def add_child_chunks(chunks: List[Dict]):
    """Добавляет дочерние чанки для поиска."""
    model = get_embedding_model()
    collection = get_chroma_collection()
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    
    ids = []
    metadatas = []
    documents = []
    
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{chunk['book_id']}_child_{chunk['chunk_index']}"
        ids.append(chunk_id)
        metadatas.append({
            'book_id': chunk['book_id'],
            'parent_id': chunk['parent_id'],
            'type': 'child'
        })
        documents.append(chunk['text'])
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def add_parent_documents(parents: List[Dict]):
    """Добавляет родительские документы для возврата."""
    model = get_embedding_model()
    collection = get_parent_collection()
    
    texts = [p['text'] for p in parents]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    
    ids = [p['parent_id'] for p in parents]
    metadatas = [{
        'book_id': p['book_id'],
        'chapter': p['chapter'],
        'type': 'parent'
    } for p in parents]
    documents = [p['text'] for p in parents]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def search_with_parents(query: str, n_results: int = 5) -> List[Dict]:
    """
    Поиск с возвратом родительских документов.
    """
    # 1. Ищем по дочерним чанкам
    child_results = search(query, n_results=n_results * 2)  # берём с запасом

    if not child_results:
        return []

    # 2. Собираем уникальные parent_id
    parent_ids = set()
    for r in child_results:
        if 'parent_id' in r:
            parent_ids.add(r['parent_id'])

    if not parent_ids:
        return []

    # 3. Получаем родительские документы из коллекции
    parent_collection = get_parent_collection()
    parent_results = parent_collection.get(
        ids=list(parent_ids),
        include=["documents", "metadatas"]
    )

    # 4. Формируем результат
    output = []
    for i in range(len(parent_results['ids'])):
        output.append({
            'book_id': parent_results['metadatas'][i]['book_id'],
            'chapter': parent_results['metadatas'][i]['chapter'],
            'parent_id': parent_results['ids'][i],
            'text': parent_results['documents'][i],
            'score': 1.0  # Можно рассчитать среднюю оценку от дочерних
        })

    return output[:n_results]

def add_chunks(chunks: List[Dict]):
    """Добавляет список чанков в векторную базу."""
    if not chunks:
        return

    model = get_embedding_model()
    collection = get_chroma_collection()

    # Подготавливаем данные
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    # Собираем тексты
    texts = [chunk['text'] for chunk in chunks]

    # Генерируем эмбеддинги пачкой
    print("Генерация эмбеддингов...")
    embeddings_list = model.encode(texts, show_progress_bar=True).tolist()

    for idx, chunk in enumerate(tqdm(chunks, desc="Подготовка данных")):
        # Уникальный ID: book_id + _ + chunk_index
        chunk_id = f"{chunk['book_id']}_{chunk['chunk_index']}"
        ids.append(chunk_id)
        embeddings.append(embeddings_list[idx])
        metadatas.append({
            "book_id": chunk['book_id'],
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
    all_chunks = get_all_chunks()
    build_bm25_index(all_chunks)


def search(query: str, n_results: int = 5) -> List[Dict]:
    if not query.strip():
        return []
    model = get_embedding_model()
    collection = get_chroma_collection()

    query_emb = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )

    output = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            score = 1 - distance
            meta = results['metadatas'][0][i]
            item = {
                'book_id': meta['book_id'],
                'chunk_index': meta['chunk_index'],
                'text': results['documents'][0][i],
                'score': score
            }
            # Если есть parent_id, добавляем
            if 'parent_id' in meta:
                item['parent_id'] = meta['parent_id']
            output.append(item)
    return output

def get_all_books() -> list[str]:
    """Возвращает списоr книг, присутствующих в векторной базе."""
    collection = get_chroma_collection()
    # Получаем метаданные всех чанков
    all_data = collection.get(include=["metadatas"])
    if not all_data['metadatas']:
        return []
    # Извлекаем book_id и оставляем только уникальные
    book_ids = set(meta['book_id'] for meta in all_data['metadatas'])
    return sorted(book_ids)

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

def get_all_chunks() -> List[Dict]:
    """Извлекает все чанки из векторной базы и возвращает в виде списка словарей."""
    collection = get_chroma_collection()
    # Получаем все записи
    all_data = collection.get(include=["documents", "metadatas"])
    chunks = []
    for i in range(len(all_data['ids'])):
        chunks.append({
            'book_id': all_data['metadatas'][i]['book_id'],
            'chunk_index': all_data['metadatas'][i]['chunk_index'],
            'text': all_data['documents'][i]
        })
    return chunks



def hybrid_search(query: str, n_results: int = 5, alpha: float = 0.5) -> List[Dict]:
    """
    Гибридный поиск: объединяет векторный поиск и BM25.
    """
    # 1. Векторный поиск
    vector_results = search(query, n_results=n_results*2)

    # 2. BM25 поиск 
    bm25_results = []
    if _bm25_index is not None and _bm25_chunks:
        query_tokens = word_tokenize(query, language='russian')
        scores = _bm25_index.get_scores(query_tokens)
        # Сортируем
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results*2]
        max_score = max(scores) if scores else 1
        if max_score == 0:
            max_score = 1
        for idx in top_indices:
            chunk = _bm25_chunks[idx].copy()
            chunk['score'] = scores[idx] / max_score
            bm25_results.append(chunk)

    # 3. Объединяем результаты
    combined = {}
    # Добавляем векторные результаты
    for r in vector_results:
        key = (r['book_id'], r['chunk_index'])
        combined[key] = {'vector': r['score'], 'bm25': 0.0, 'data': r}
    # Добавляем BM25 результаты
    for r in bm25_results:
        key = (r['book_id'], r['chunk_index'])
        if key in combined:
            combined[key]['bm25'] = r['score']
        else:
            combined[key] = {'vector': 0.0, 'bm25': r['score'], 'data': r}

    hybrid_list = []
    for key, scores in combined.items():
        hybrid_score = alpha * scores['vector'] + (1 - alpha) * scores['bm25']
        if hybrid_score >= MIN_SCORE:
            item = scores['data'].copy()
            item['score'] = hybrid_score
            hybrid_list.append(item)

    # Сортируем по убыванию гибридной оценки и берём топ-n_results
    hybrid_list.sort(key=lambda x: x['score'], reverse=True)
    return hybrid_list[:n_results]

def expanded_search(query: str, n_results: int = 5) -> List[Dict]:
    """
    Выполняет поиск с расширением запроса.
    """
    # Получаем варианты запросов
    queries = expand_query(query, num_expansions=2)
    print(f"Expanded queries: {queries}")  # отладка

    # Собираем результаты для каждого запроса
    all_results = []
    for q in queries:
        results = hybrid_search(q, n_results=n_results * 2)  # берём с запасом
        all_results.extend(results)

    # Убираем дубликаты по ID чанка (book_id + chunk_index)
    unique = {}
    for r in all_results:
        key = f"{r['book_id']}_{r['chunk_index']}"
        if key not in unique or r['score'] > unique[key]['score']:
            unique[key] = r

    # Сортируем по убыванию скоров
    sorted_results = sorted(unique.values(), key=lambda x: x['score'], reverse=True)
    return sorted_results[:n_results]

def clear_database():
    """Полностью очищает коллекцию книг в базе данных."""
    collection = get_chroma_collection()
    # Получаем все ID
    all_data = collection.get(include=[]) 
    ids = all_data.get('ids', [])
    if ids:
        collection.delete(ids=ids)
        print(f"Удалено {len(ids)} записей из коллекции.")
    else:
        print("Коллекция уже пуста.")

def load_standard_books():
    """Загружает все книги из папки standard_books и индексирует их."""


    std_dir = "standard_books"
    if not os.path.exists(std_dir):
        os.makedirs(std_dir)
        print(f"Папка {std_dir} создана. Положите в неё книги.")
        return

    files = [f for f in os.listdir(std_dir) if f.endswith('.txt')]
    if not files:
        print("В папке standard_books нет .txt файлов.")
        return

    for filename in files:
        filepath = os.path.join(std_dir, filename)
        book_id = os.path.splitext(filename)[0]
        print(f"Обработка {filename}...")
        chunks = process_book(filepath, book_id)
        if chunks:
            add_chunks(chunks)  
    print("Загрузка стандартных книг завершена.")