from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Базовая директория проекта (rag-chatbot/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Папка с PDF
DOCS_DIR = BASE_DIR / "data" / "docs"

# Папка для Chroma
VECTORSTORE_DIR = BASE_DIR / "data" / "chroma_db"

# Настройки чанкинга
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
