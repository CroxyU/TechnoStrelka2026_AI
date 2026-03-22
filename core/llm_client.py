import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY не найден в .env")

# Инициализируем клиент (один раз)
_client = None

def get_groq_client():
    """Ленивая инициализация клиента Groq"""
    global _client
    if _client is None:
        _client = Groq(api_key=API_KEY)
    return _client

def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    """
    Отправляет запрос в Groq с вопросом и контекстом.
    Возвращает сгенерированный ответ.
    """
    if not context_chunks:
        return "Недостаточно информации для ответа."

    # Формируем контекст из текстов чанков
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])

    # Промпт, который заставляет модель отвечать только по контексту
    system_prompt = """Ты — помощник, который отвечает на вопросы, используя только предоставленный контекст из книг.
Если ответа нет в контексте, скажи: «Я не знаю ответа на этот вопрос, так как в загруженных книгах нет такой информации.»

Контекст:
{context}

Вопрос: {question}

Ответ:"""

    user_prompt = system_prompt.format(context=context, question=question)

    try:
        client = get_groq_client()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Ты — полезный ассистент, отвечающий только на основе предоставленного контекста."
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model="llama-3.3-70b-versatile",  # отличная модель для русского языка
            temperature=0.3,  # низкая температура для более точных ответов
            max_tokens=500,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка при обращении к Groq: {e}")
        return f"Произошла ошибка при генерации ответа: {e}"

def expand_query(query: str, num_expansions: int = 5) -> list[str]:
    """
    Генерирует альтернативные формулировки запроса с помощью Groq.
    Возвращает список, включающий исходный запрос.
    """
    if not query.strip():
        return [query]

    client = Groq(api_key=API_KEY)

    prompt = f"""Придумай {num_expansions} альтернативных формулировок для поискового запроса. 
    Запрос: "{query}"
    
    Требования:
    - Формулировки должны быть на том же языке, что и запрос.
    - Они могут включать синонимы, перефразирование, другие формы слов, более развёрнутые или краткие варианты.
    - Не используй нумерацию, просто перечисли каждую формулировку с новой строки.
    - Не добавляй пояснений, только сами формулировки.
    
    Альтернативные формулировки:"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", 
            temperature=0.7,
            max_tokens=200
        )
        text = response.choices[0].message.content.strip()
        # Разбиваем на строки, убираем пустые
        alternatives = [line.strip() for line in text.split('\n') if line.strip()]
        # Ограничиваем количество
        alternatives = alternatives[:num_expansions]
        # Возвращаем исходный запрос + альтернативы
        return [query] + alternatives
    except Exception as e:
        print(f"Ошибка при расширении запроса: {e}")
        return [query]  # при ошибке используем только исходный запрос