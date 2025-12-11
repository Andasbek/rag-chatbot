from .loaders import load_pdf
from .splitter import split_documents
from .vectorstore import build_vectorstore
from .rag_chain import create_rag_chain, ask_question


def prepare_index(pdf_name: str):
    """
    Полный цикл индексации PDF: загрузка -> чанкинг -> Chroma.
    """
    print(f"[MAIN] === НАЧАЛО ИНДЕКСАЦИИ ДЛЯ ФАЙЛА: {pdf_name} ===")
    docs = load_pdf(pdf_name)
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("[MAIN] === ИНДЕКСАЦИЯ ЗАВЕРШЕНА ===")


def chat():
    """
    Простой CLI-чат с RAG-ботом.
    """
    chain, retriever = create_rag_chain()
    print("RAG-бот запущен. Введите вопрос (или 'exit'):")

    while True:
        q = input(">> ")
        if q.lower() in ["exit", "quit", "q"]:
            break

        answer, sources = ask_question(chain, retriever, q)

        print("\nОтвет:")
        print(answer)

        print("\nИсточники (страницы PDF):")
        for s in sources:
            print("-", s.metadata.get("page", "page?"))
        print("-" * 40)


if __name__ == "__main__":
    # ДЛЯ ОТЛАДКИ: каждый запуск пересобираем индекс и сразу запускаем чат
    prepare_index("sample.pdf")
    chat()

    # Когда убедишься, что индекс создаётся, можно будет сделать так:
    # 1) один раз вызвать prepare_index("sample.pdf")
    # 2) потом закомментировать prepare_index и оставить только chat()
