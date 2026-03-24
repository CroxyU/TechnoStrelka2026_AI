import os
import re
from typing import List, Dict, Optional
import nltk

# Скачиваем токенизатор для предложений (если не скачан)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def load_text_file(filepath: str) -> str:
    encodings = ['utf-8', 'windows-1251', 'koi8-r', 'latin-1']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось прочитать файл {filepath}")


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def split_into_chapters(text: str) -> List[Dict[str, str]]:
    """
    Пытается разбить текст на главы по маркерам вида 'Глава X', 'Chapter X' и т.п.
    Если маркеры не найдены, возвращает одну главу с заголовком 'Весь текст'.
    """
    # Паттерны для поиска заголовков глав (рус/англ)
    patterns = [
        r'^(глава|ГЛАВА)\s+(\d+|[IVXLCDM]+)',
        r'^(часть|ЧАСТЬ)\s+(\d+|[IVXLCDM]+)',
        r'^(chapter|CHAPTER)\s+(\d+)',
        r'^(part|PART)\s+(\d+)',
        r'^\d+\.',                       # "1. " в начале строки
        r'^[IVXLCDM]+\.',                # "I. " в начале строки
        r'^[А-ЯЁ]{4,}$'                  # строка из заглавных букв (возможный заголовок)
    ]
    combined = '|'.join(patterns)
    chapter_pattern = re.compile(combined, re.IGNORECASE | re.MULTILINE)

    matches = list(chapter_pattern.finditer(text))

    if not matches:
        return [{'title': 'Весь текст', 'content': text}]

    chapters = []
    prev_end = 0
    for i, match in enumerate(matches):
        start, end = match.span()
        if i > 0:
            content = text[prev_end:start].strip()
            chapters.append({'title': matches[i-1].group(0).strip(), 'content': content})
        prev_end = end

    if prev_end < len(text):
        content = text[prev_end:].strip()
        chapters.append({'title': matches[-1].group(0).strip(), 'content': content})

    return chapters


def split_into_sentence_chunks(text: str, chunk_size: int = 350, overlap_sentences: int = 1) -> List[str]:
    """
    Разбивает текст на чанки, состоящие из целых предложений.
    chunk_size – желаемая длина чанка в символах.
    overlap_sentences – количество предложений перекрытия.
    """
    sentences = nltk.sent_tokenize(text, language='russian')
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current_chunk:
            # Сохраняем текущий чанк
            chunks.append(' '.join(current_chunk))
            # Начинаем новый чанк с перекрытием
            if overlap_sentences > 0:
                overlap_start = max(0, len(current_chunk) - overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_len = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_len = 0
        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def process_book_with_parents(filepath: str, book_id: str,
                              parent_size: int = 1500,
                              child_size: int = 300) -> tuple[list[dict], list[dict]]:
    """
    Обрабатывает книгу и создаёт иерархию:
    - Родительские документы (главы или большие блоки)
    - Дочерние чанки для поиска

    Возвращает (child_chunks, parent_documents)
    """
    text = load_text_file(filepath)
    text = clean_text(text)

    # Сначала получаем главы (родительские документы)
    chapters = split_into_chapters(text)

    child_chunks = []
    parent_documents = []
    parent_counter = 0
    child_counter = 0

    for chapter in chapters:
        chapter_title = chapter['title']
        chapter_content = chapter['content']

        # Если глава слишком большая, разбиваем её на родительские блоки
        if len(chapter_content) > parent_size:
            # Разбиваем главу на блоки подходящего размера
            parent_blocks = split_into_sentence_chunks(
                chapter_content,
                chunk_size=parent_size,
                overlap_sentences=2
            )
            for block in parent_blocks:
                parent_id = f"{book_id}_parent_{parent_counter}"
                parent_documents.append({
                    'parent_id': parent_id,
                    'book_id': book_id,
                    'chapter': chapter_title,
                    'text': block,
                    'type': 'parent'
                })

                # Разбиваем родительский блок на дочерние чанки
                child_blocks = split_into_sentence_chunks(
                    block,
                    chunk_size=child_size,
                    overlap_sentences=1
                )
                for child_text in child_blocks:
                    child_chunks.append({
                        'book_id': book_id,
                        'chapter': chapter_title,
                        'parent_id': parent_id,
                        'chunk_index': child_counter,
                        'text': child_text
                    })
                    child_counter += 1
                parent_counter += 1
        else:
            # Глава помещается в один родительский документ
            parent_id = f"{book_id}_parent_{parent_counter}"
            parent_documents.append({
                'parent_id': parent_id,
                'book_id': book_id,
                'chapter': chapter_title,
                'text': chapter_content,
                'type': 'parent'
            })

            # Разбиваем главу на дочерние чанки
            child_blocks = split_into_sentence_chunks(
                chapter_content,
                chunk_size=child_size,
                overlap_sentences=1
            )
            for child_text in child_blocks:
                child_chunks.append({
                    'book_id': book_id,
                    'chapter': chapter_title,
                    'parent_id': parent_id,
                    'chunk_index': child_counter,
                    'text': child_text
                })
                child_counter += 1
            parent_counter += 1

    return child_chunks, parent_documents


def process_book(filepath: str, book_id: str) -> List[Dict]:
    """Простая обработка книги без иерархии (только чанки по предложениям)."""
    text = load_text_file(filepath)
    text = clean_text(text)

    chunks = []
    chunk_counter = 0
    ch_chunks = split_into_sentence_chunks(text)

    for i, chunk_text in enumerate(ch_chunks):
        chunks.append({
            'book_id': book_id,
            'chunk_index': chunk_counter,
            'text': chunk_text
        })
        chunk_counter += 1

    return chunks