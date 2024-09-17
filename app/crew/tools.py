import requests
import os
import json
import logging
import requests
from requests.exceptions import MissingSchema

from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from crewai_tools.tools import FileReadTool
from urllib.parse import urlencode
from crew.errors import NoGazettesFoundError, CityNotFoundError

json_file_path = os.path.join(os.path.dirname(__file__), "..", "ibge_cities_code.json")
with open(json_file_path, "r", encoding="utf-8") as f:
    cities = json.load(f)

querido_diario_advanced_search_context_tool = FileReadTool(
    file_path="app/querido_diario_search_context.txt",
    description="A tool designed to efficiently read and interpret search operators context to construct the subject",
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


class QueridoDiarioTools:
    @tool("Fetch Querido Diario API")
    def querido_diario_fetch(
        subject: str,
        city: str = None,
        published_since: str = None,
        published_until: str = None,
    ):
        """Fetchs data from querido diario API to search about municipal gazettes"""
        try:
            api_url = "https://queridodiario.ok.org.br/api/gazettes"

            query_params = {"querystring": subject, "sort_by": "relevance"}

            if city and city in cities:
                query_params["territory_ids"] = cities[city]
            else:
                raise CityNotFoundError()

            if published_since and published_since.lower() != "none":
                query_params["published_since"] = published_since
            if published_until and published_until.lower() != "none":
                query_params["published_until"] = published_until

            query_string = urlencode(query_params)
            response = requests.get(f"{api_url}?{query_string}")
            gazettes = response.json().get("gazettes", [])

            if not gazettes:
                raise NoGazettesFoundError()

            # TODO: Return and handle the 5 most relevant public gazettes
            return gazettes[0]

        except NoGazettesFoundError as e:
            print("e ", e)
            logging.error(f"Error in gazette_data_retrieval no gazettes found: {e}")
            return {"error": "No public gazettes found"}
        except CityNotFoundError as e:
            logging.error(f"Error in gazette_data_retrieval: {e}")
            return {"error": "City not found in our database"}
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"error": "An unexpected error occurred."}

    @tool("Search in municipal gazette")
    def gazette_search_context(claim: str, url: str, questions: dict):
        """Search data from municipal gazettes to get relevant documents"""
        try:
            embedding = OpenAIEmbeddings()
            loader = WebBaseLoader(url)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators="\n",
                chunk_size=1000,
                chunk_overlap=200,
            )

            documents = text_splitter.split_documents(docs)
            retriever = FAISS.from_documents(documents, embedding).as_retriever(
                search_type="similarity", search_kwargs={"k": 10}
            )
            return retriever.invoke(claim)
        except MissingSchema:
            logging.error(
                f"Error invalid URL: The URL '{url}' is missing a scheme (e.g., 'http://' or 'https://'). Please provide a valid URL."
            )
            return {"error": "Invalid URL provided."}
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP error occurred: {err}")
            return {"error": "HTTP error occurred."}
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"error": "An unexpected error occurred."}

    # TODO: Implement querido diario glossario tool
    @tool("Search Context in Querido Diario Glossario")
    def querido_diario_glossario_tool(claim):
        """Use to find glossario context information"""
        embedding = OpenAIEmbeddings()
        loader = DocumentLoader("./querido_diario_glossario_context.txt")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=200,
        )

        documents = text_splitter.split_documents(docs)

        retriever = FAISS.from_documents(documents, embedding).as_retriever()
        return retriever.invoke(claim)
