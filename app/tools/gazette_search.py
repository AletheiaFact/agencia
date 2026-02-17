import logging
import os
from typing import Iterator, Optional

import requests
from requests.exceptions import MissingSchema
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader(BaseLoader):
    """Document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1


@tool("search_municipal_gazette")
def gazette_search_context(claim: str, url: str, questions: Optional[str] = None):
    """Search data from municipal gazettes to get relevant documents."""
    logger.info("[search_municipal_gazette] claim=%s url=%s questions=%s", claim[:80], url[:120] if url else "N/A", questions)
    try:
        embedding = OpenAIEmbeddings()
        loader = WebBaseLoader(url)
        docs = loader.load()
        logger.info("[search_municipal_gazette] Loaded %d documents from URL", len(docs))

        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=200,
        )

        documents = text_splitter.split_documents(docs)
        logger.info("[search_municipal_gazette] Split into %d chunks, building FAISS index", len(documents))
        retriever = FAISS.from_documents(documents, embedding).as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        )
        results = retriever.invoke(claim)
        logger.info("[search_municipal_gazette] FAISS returned %d results", len(results))
        return results
    except MissingSchema:
        logger.error("[search_municipal_gazette] Invalid URL: '%s' missing scheme", url)
        return {"error": "Invalid URL provided."}
    except requests.exceptions.HTTPError as err:
        logger.error("[search_municipal_gazette] HTTP error: %s", err)
        return {"error": "HTTP error occurred."}
    except Exception as e:
        logger.error("[search_municipal_gazette] Unexpected error: %s", e)
        return {"error": "An unexpected error occurred."}


@tool("search_querido_diario_glossario")
def querido_diario_glossario_tool(claim: str):
    """Use to find glossario context information."""
    logger.info("[search_querido_diario_glossario] claim=%s", claim[:80])
    glossary_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "querido_diario_glossario_context.txt"
    )
    embedding = OpenAIEmbeddings()
    loader = DocumentLoader(glossary_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = text_splitter.split_documents(docs)
    retriever = FAISS.from_documents(documents, embedding).as_retriever()
    results = retriever.invoke(claim)
    logger.info("[search_querido_diario_glossario] Returned %d results", len(results))
    return results
