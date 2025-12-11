from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from .config import DOCS_DIR


def load_pdf(file_name: str):
    """
    Загружает PDF из папки data/docs и возвращает список Document.
    """
    path = Path(DOCS_DIR) / file_name
    print(f"[LOAD_PDF] Пытаюсь загрузить файл: {path}")

    if not path.exists():
        raise FileNotFoundError(f"PDF не найден: {path}")

    loader = PyPDFLoader(str(path))
    docs = loader.load()
    print(f"[LOAD_PDF] Загружено страниц/документов: {len(docs)}")
    return docs
