from core.vector_store import search

if __name__ == "__main__":
    query = input("Введите поисковый запрос: ")
    results = search(query, n_results=3)
    if not results:
        print("Ничего не найдено.")
    else:
        for r in results:
            print(f"\n--- [Книга: {r['book_id']}, Глава: {r['chapter']}, Сходство: {r['score']:.3f}] ---")
            print(r['text'][:300] + "...")