from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .config import VECTORSTORE_DIR


def build_vectorstore(chunks):
    """
    Создает и сохраняет векторное хранилище (Chroma) из чанков.
    """
    print(f"[VECTORSTORE] Создаю Chroma в {VECTORSTORE_DIR}")
    embeddings = OpenAIEmbeddings()
    vs = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )
    # vs.persist()
    print("[VECTORSTORE] Векторное хранилище создано и сохранено.")
    return vs


def load_vectorstore():
    """
    Загружает существующее векторное хранилище Chroma.
    """
    print(f"[VECTORSTORE] Загружаю Chroma из {VECTORSTORE_DIR}")
    embeddings = OpenAIEmbeddings()
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )
    # небольшая проверка через внутреннюю коллекцию
    try:
        count = vs._collection.count()
        print(f"[VECTORSTORE] Количество записей в коллекции: {count}")
    except Exception as e:
        print(f"[VECTORSTORE] Не удалось получить count: {e}")
    return vs
