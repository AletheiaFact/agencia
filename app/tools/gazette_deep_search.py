"""Multi-document FAISS search tool for gazette full-text analysis.

Replaces the single-document gazette_search_context tool with support for:
- Multiple gazette URLs downloaded and indexed together
- Single FAISS index across all documents
- Higher default top_k for richer evidence extraction
"""

import logging
from typing import Optional

import requests
from requests.exceptions import MissingSchema
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _download_gazette_text(url: str) -> Optional[str]:
    """Download gazette text from a txt_url."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error("[gazette_deep_search] Failed to download %s: %s", url[:120], e)
        return None


@tool("gazette_deep_search")
def gazette_deep_search(
    claim: str,
    urls: list[str],
    questions: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 10,
) -> str:
    """Download and search across multiple gazette documents using FAISS.

    Downloads full text for each URL, builds a unified FAISS index across
    all documents, and returns the most relevant passages for the claim.

    Args:
        claim: The fact-checking claim to search for in the gazettes.
        urls: List of gazette txt_urls to download and index.
        questions: Optional related questions to refine the search context.
        chunk_size: Size of text chunks for splitting (default 1000).
        chunk_overlap: Overlap between chunks (default 200).
        top_k: Number of top results to return from FAISS (default 15).
    """
    logger.info(
        "[gazette_deep_search] claim='%s' urls=%d questions=%s top_k=%d",
        claim[:80], len(urls), bool(questions), top_k,
    )

    MAX_CHARS_PER_GAZETTE = 80_000  # ~20K tokens per gazette, keeps total manageable

    all_documents: list[Document] = []
    downloaded_count = 0

    for url in urls:
        text = _download_gazette_text(url)
        if text:
            downloaded_count += 1
            if len(text) > MAX_CHARS_PER_GAZETTE:
                logger.info(
                    "[gazette_deep_search] Truncating %s from %d to %d chars",
                    url[:80], len(text), MAX_CHARS_PER_GAZETTE,
                )
                text = text[:MAX_CHARS_PER_GAZETTE]
            all_documents.append(Document(
                page_content=text,
                metadata={"source": url},
            ))
            logger.info(
                "[gazette_deep_search] Downloaded %s (%d chars)",
                url[:80], len(text),
            )

    if not all_documents:
        logger.warning("[gazette_deep_search] No documents could be downloaded")
        return "Error: No gazette documents could be downloaded from the provided URLs."

    logger.info(
        "[gazette_deep_search] Downloaded %d/%d gazettes, splitting into chunks",
        downloaded_count, len(urls),
    )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(all_documents)
    logger.info("[gazette_deep_search] Split into %d chunks, building FAISS index", len(chunks))

    if not chunks:
        return "Error: Documents were downloaded but produced no text chunks."

    # Build FAISS index with retry — if embedding fails due to token limits,
    # reduce chunk count by half and retry (up to 3 times)
    embedding = OpenAIEmbeddings()
    max_retries = 3
    current_chunks = chunks
    vectorstore = None

    for attempt in range(max_retries + 1):
        try:
            vectorstore = FAISS.from_documents(current_chunks, embedding)
            break
        except Exception as exc:
            msg = str(exc).lower()
            is_token_error = "token" in msg and (
                "max" in msg or "limit" in msg or "exceed" in msg or "requested" in msg
            )
            if not is_token_error or attempt >= max_retries:
                raise
            # Halve the chunks and retry
            new_count = max(len(current_chunks) // 2, 10)
            logger.warning(
                "[gazette_deep_search] Embedding failed (attempt %d/%d, %d chunks). "
                "Reducing to %d chunks and retrying.",
                attempt + 1, max_retries, len(current_chunks), new_count,
            )
            current_chunks = current_chunks[:new_count]

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": min(top_k, len(current_chunks))},
    )

    search_query = claim
    if questions:
        search_query = f"{claim}\n\nRelated questions: {questions}"

    results = retriever.invoke(search_query)
    logger.info("[gazette_deep_search] FAISS returned %d results", len(results))

    # Format results with source attribution
    formatted_parts: list[str] = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        formatted_parts.append(f"[Result {i} — source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_parts)
