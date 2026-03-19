
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

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)  
    return text.strip()

"""Разбиваем на чанки"""

import nltk
nltk.download('punkt_tab')  # один раз

def chunk_by_sentences(text: str, chunk_size: int = 350, overlap_sentences: int = 1):
    sentences = nltk.sent_tokenize(text)
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

"""#Обработка"""

def process_book(filepath: str, book_id: str) -> List[Dict]:

    text = load_text_file(filepath)
    text = clean_text(text)



    chunks = []
    chunk_counter = 0


    
    ch_chunks = chunk_by_sentences(text)

    for i, chunk_text in enumerate(ch_chunks):
        chunks.append({
            'book_id': book_id,
            'chunk_index': chunk_counter,
            'text': chunk_text
        })
        chunk_counter += 1

    return chunks


