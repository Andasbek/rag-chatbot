from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .vectorstore import load_vectorstore


def format_docs(docs):
    """
    Собирает найденные документы в один текст для контекста.
    """
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        parts.append(f"Страница {page}:\n{d.page_content}")
    return "\n\n".join(parts)


def create_rag_chain():
    """
    Создаёт RAG-цепочку в стиле LangChain 0.3+.
    Возвращает (chain, retriever).
    """
    llm = ChatOpenAI(
        model="gpt-4o",     # можно gpt-4o-mini
        temperature=0
    )

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "Ты — помощник, который ОТВЕЧАЕТ ИСКЛЮЧИТЕЛЬНО на основе содержимого "
        "переданного PDF-документа.\n"
        "У тебя НЕТ собственных знаний, кроме текста в контексте.\n\n"
        "Правила:\n"
        "1) Используй только предоставленный контекст.\n"
        "2) Если в контексте НЕТ достаточной информации для ответа на вопрос,\n"
        "   напиши дословно: «В документе нет информации, достаточной для ответа на этот вопрос.»\n"
        "3) Не придумывай факты и не используй внешние знания.\n"
        "4) Отвечай на том же языке, на котором задан вопрос.\n\n"
        "Контекст:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def ask_question(chain, retriever, query: str):
    """
    Получаем ответ + источники.
    """
    answer = chain.invoke(query)
    sources = retriever.invoke(query)  # список Document

    print("\n[DEBUG] Извлечённые фрагменты из документа:")
    if not sources:
        print("[DEBUG] Ретривер не нашёл ни одного фрагмента.")
    for doc in sources:
        print("---- страница:", doc.metadata.get("page"))
        print(doc.page_content[:400], "...\n")

    return answer, sources
