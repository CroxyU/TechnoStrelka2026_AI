# -*- coding: utf-8 -*-

import os
import re
from typing import List, Dict, Optional

def load_text_file(filepath: str) -> str:
    encodings = ['utf-8', 'windows-1251', 'koi8-r', 'latin-1']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось прочитать файл {filepath}")

"""Очистка текста"""

def clean_text(text: str) -> str:

    # Заменяем табуляции и множественные пробелы на один пробел
    text = re.sub(r'\s+', ' ', text)
    # Восстанавливаем переносы строк там, где они реально нужны (например, после точки)
    # Но пока оставим как есть, просто уберём слишком длинные последовательности пустых строк
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Двойные переносы оставляем как разделители абзацев
    return text.strip()

"""Разбиваем на чанки"""

import nltk
nltk.download('punkt_tab')  # один раз

def chunk_by_sentences(text: str, chunk_size: int = 200, overlap_sentences: int = 1):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current_chunk:
            # Сохраняем текущий чанк
            chunks.append(' '.join(current_chunk))
            # Начинаем новый чанк с перекрытием: оставляем последние overlap_sentences предложений
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

"""#Сохраняем положение в тексте"""

def split_into_chapters(text: str) -> List[Dict[str, str]]:

    # Регулярное выражение для поиска заголовков глав (рус/англ)
    chapter_pattern = re.compile(r'^(глава|chapter)\s+([^\n]+)', re.IGNORECASE | re.MULTILINE)
    matches = list(chapter_pattern.finditer(text))

    if not matches:
        # Нет глав — возвращаем весь текст как одну главу
        return [{'title': 'Весь текст', 'content': text}]

    chapters = []
    prev_end = 0
    for i, match in enumerate(matches):
        start, end = match.span()
        # Текст до текущей главы — предыдущая глава (кроме первой)
        if i > 0:
            chapter_content = text[prev_end:start].strip()
            chapters.append({'title': matches[i-1].group(0).strip(), 'content': chapter_content})
        prev_end = end

    # Последний кусок после последнего заголовка
    if prev_end < len(text):
        chapters.append({'title': matches[-1].group(0).strip(), 'content': text[prev_end:].strip()})

    return chapters

"""#Обработка"""

def process_book(filepath: str, book_id: str) -> List[Dict]:

    text = load_text_file(filepath)
    text = clean_text(text)

    chapters = split_into_chapters(text)

    chunks = []
    chunk_counter = 0
    for chapter in chapters:
        chapter_title = chapter['title']
        chapter_content = chapter['content']

        # Разбиваем содержание главы на чанки
        chapter_chunks = chunk_by_sentences(chapter_content)

        for i, chunk_text in enumerate(chapter_chunks):
            chunks.append({
                'book_id': book_id,
                'chapter': chapter_title,
                'chunk_index': chunk_counter,  # сквозная нумерация по всей книге
                'text': chunk_text
            })
            chunk_counter += 1

    return chunks


