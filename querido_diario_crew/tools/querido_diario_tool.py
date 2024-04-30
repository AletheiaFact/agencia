import requests
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from querido_diario_crew.tools.document_loader_tool import DocumentLoader
from langchain_community.document_loaders import WebBaseLoader

#TODO: Get cities from IBGE
cities = {
    "São José dos Campos": "3549904",
    "São José dos Mambos": "0000000"
}

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