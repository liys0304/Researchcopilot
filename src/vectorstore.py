import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.config import DOCS_DIR, CHROMA_DIR
from src.llm import get_embedding_model


def load_all_docs():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DOCS_DIR, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs


def build_vectorstore():
    docs = load_all_docs()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=get_embedding_model(),
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    return vectordb


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embedding_model()
    )


def similarity_search(query: str, k: int = 3):
    db = get_vectorstore()
    docs = db.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]