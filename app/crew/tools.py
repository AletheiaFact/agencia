import requests
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from crewai_tools.tools import FileReadTool

#TODO: Get cities from IBGE
cities = {
    "São José dos Campos": "3549904",
    "São José dos Mambos": "0000000"
}

querido_diario_advanced_search_context_tool = FileReadTool(
	file_path='app/querido_diario_search_context.txt',
	description="A tool designed to efficiently read and interpret search operators context to construct the subject"
)

class DocumentLoader(BaseLoader):
    """Document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path"""
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader that reads a file line by line."""
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1

class QueridoDiarioTools():
    @tool("Fetch Querido Diario API")
    def querido_diario_fetch(city: str, subject: str, published_since: str, published_until: str):
        """Fetchs data from querido diario API to search about municipal gazettes"""
        territory_id = cities[city]
        api_url = "https://queridodiario.ok.org.br/api/gazettes"
        query = f"querystring={subject}&territory_ids={territory_id}&published_since={published_since}&published_until={published_until}&sort_by=relevance"
        response = requests.get(f"{api_url}?{query}")
        gazettes = response.json()["gazettes"]
        return gazettes[0]
  
    @tool("Search in municipal gazette")
    def gazette_search_context(claim: str, url: str, subject: str):
        """Search data from municipal gazettes to get relevant documents"""
        embedding = OpenAIEmbeddings()
        loader = WebBaseLoader(url)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=200,
        )

        documents = text_splitter.split_documents(docs)

        retriever = FAISS.from_documents(documents, embedding).as_retriever()
        return retriever.invoke(input=claim + subject)

    @tool("Search Context in Querido Diario Glossario")
    def querido_diario_glossario_tool(claim):
        """Use to find glossario context information"""
        embedding = OpenAIEmbeddings()
        loader = DocumentLoader('./querido_diario_glossario_context.txt')
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=200,
        )

        documents = text_splitter.split_documents(docs)

        retriever = FAISS.from_documents(documents, embedding).as_retriever()
        return retriever.invoke(claim)