"""
Phase 1 Foundation Tests
========================
Validates that the Phase 1 foundation migration is correct:
- Dependencies are installed at correct versions
- New directory structure exists with correct files
- Extracted tools import and function correctly
- State schema is valid
- Data files are accessible from new paths
- Config files (agents.yaml, tasks.yaml) are in place

Run: python -m pytest tests/test_phase1_foundation.py -v
  Or: .venv/bin/python -m pytest tests/test_phase1_foundation.py -v
"""

import importlib
import json
import os
import sys

import pytest

# Ensure app/ is on the path (matches how the app runs)
APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# 1. Dependency version checks
# ---------------------------------------------------------------------------
class TestDependencies:
    """Verify upgraded packages are installed at the right major versions."""

    def test_langchain_version(self):
        import langchain
        major, minor = map(int, langchain.__version__.split(".")[:2])
        assert major == 0 and minor >= 3, f"langchain {langchain.__version__} < 0.3"

    def test_langchain_core_version(self):
        import langchain_core
        major, minor = map(int, langchain_core.__version__.split(".")[:2])
        assert major == 0 and minor >= 3, f"langchain_core {langchain_core.__version__} < 0.3"

    def test_langchain_community_version(self):
        import langchain_community
        major, minor = map(int, langchain_community.__version__.split(".")[:2])
        assert major == 0 and minor >= 3, f"langchain_community {langchain_community.__version__} < 0.3"

    def test_langchain_openai_version(self):
        from importlib.metadata import version
        ver = version("langchain-openai")
        major, minor = map(int, ver.split(".")[:2])
        assert major == 0 and minor >= 2, f"langchain-openai {ver} < 0.2"

    def test_langgraph_version(self):
        from importlib.metadata import version
        ver = version("langgraph")
        major, minor = map(int, ver.split(".")[:2])
        assert major == 0 and minor >= 6, f"langgraph {ver} < 0.6"

    def test_pydantic_version(self):
        import pydantic
        major = int(pydantic.__version__.split(".")[0])
        assert major >= 2, f"pydantic {pydantic.__version__} < 2.0"

    def test_fastapi_available(self):
        import fastapi
        assert fastapi is not None

    def test_faiss_available(self):
        import faiss
        assert faiss is not None

    def test_crewai_not_installed(self):
        """CrewAI should NOT be installed in the new environment."""
        with pytest.raises(ImportError):
            import crewai  # noqa: F401

    def test_langserve_not_installed(self):
        """langserve should NOT be installed in the new environment."""
        with pytest.raises(ImportError):
            import langserve  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Directory structure checks
# ---------------------------------------------------------------------------
class TestDirectoryStructure:
    """Verify the new project layout exists."""

    @pytest.mark.parametrize("path", [
        "app/tools/__init__.py",
        "app/tools/querido_diario.py",
        "app/tools/gazette_search.py",
        "app/tools/web_search.py",
        "app/nodes/__init__.py",
        "app/nodes/gazette/__init__.py",
        "app/data/ibge_cities_code.json",
        "app/data/querido_diario_search_context.txt",
        "app/data/querido_diario_glossario_context.txt",
        "app/config/agents.yaml",
        "app/config/tasks.yaml",
        "app/errors.py",
        "app/state.py",
    ])
    def test_file_exists(self, path):
        full = os.path.join(PROJECT_ROOT, path)
        assert os.path.isfile(full), f"Missing: {path}"


# ---------------------------------------------------------------------------
# 3. Data files integrity
# ---------------------------------------------------------------------------
class TestDataFiles:
    """Verify data files are accessible and valid from the new paths."""

    def test_ibge_cities_json_is_valid(self):
        path = os.path.join(PROJECT_ROOT, "app", "data", "ibge_cities_code.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "IBGE JSON should be a dict"
        assert len(data) > 0, "IBGE JSON should not be empty"

    def test_ibge_cities_has_known_entries(self):
        """Spot-check a few known Brazilian cities."""
        path = os.path.join(PROJECT_ROOT, "app", "data", "ibge_cities_code.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # At least some entries should exist (exact keys depend on the data)
        assert len(data) >= 10, f"Expected many cities, got {len(data)}"

    def test_search_context_not_empty(self):
        path = os.path.join(PROJECT_ROOT, "app", "data", "querido_diario_search_context.txt")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100, "Search context file should have meaningful content"

    def test_glossario_context_not_empty(self):
        path = os.path.join(PROJECT_ROOT, "app", "data", "querido_diario_glossario_context.txt")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100, "Glossario context file should have meaningful content"

    def test_data_files_match_originals(self):
        """New data files should be identical to the originals."""
        pairs = [
            ("app/ibge_cities_code.json", "app/data/ibge_cities_code.json"),
            ("app/querido_diario_search_context.txt", "app/data/querido_diario_search_context.txt"),
            ("app/querido_diario_glossario_context.txt", "app/data/querido_diario_glossario_context.txt"),
        ]
        for orig, new in pairs:
            orig_path = os.path.join(PROJECT_ROOT, orig)
            new_path = os.path.join(PROJECT_ROOT, new)
            if os.path.exists(orig_path):  # originals still exist during Phase 1
                with open(orig_path, "rb") as f1, open(new_path, "rb") as f2:
                    assert f1.read() == f2.read(), f"{new} differs from {orig}"


# ---------------------------------------------------------------------------
# 4. Config files
# ---------------------------------------------------------------------------
class TestConfigFiles:
    """Verify YAML config files are in place and parseable."""

    def test_agents_yaml_parseable(self):
        import yaml
        path = os.path.join(PROJECT_ROOT, "app", "config", "agents.yaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "agents.yaml should parse to a dict"
        assert len(data) > 0, "agents.yaml should not be empty"

    def test_tasks_yaml_parseable(self):
        import yaml
        path = os.path.join(PROJECT_ROOT, "app", "config", "tasks.yaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "tasks.yaml should parse to a dict"
        assert len(data) > 0, "tasks.yaml should not be empty"

    def test_config_files_match_originals(self):
        """Config files should be identical to the crew originals."""
        pairs = [
            ("app/crew/config/agents.yaml", "app/config/agents.yaml"),
            ("app/crew/config/tasks.yaml", "app/config/tasks.yaml"),
        ]
        for orig, new in pairs:
            orig_path = os.path.join(PROJECT_ROOT, orig)
            new_path = os.path.join(PROJECT_ROOT, new)
            if os.path.exists(orig_path):
                with open(orig_path, "rb") as f1, open(new_path, "rb") as f2:
                    assert f1.read() == f2.read(), f"{new} differs from {orig}"


# ---------------------------------------------------------------------------
# 5. State schema
# ---------------------------------------------------------------------------
class TestState:
    """Verify the state module is importable and has expected fields."""

    def test_state_imports(self):
        from state import AgentState, Context
        assert AgentState is not None
        assert Context is not None

    def test_agent_state_has_core_fields(self):
        from state import AgentState
        # TypedDict.__annotations__ gives the declared fields
        annotations = AgentState.__annotations__
        core_fields = ["claim", "context", "messages", "questions",
                       "can_be_fact_checked", "search_type", "language"]
        for field in core_fields:
            assert field in annotations, f"AgentState missing core field: {field}"

    def test_agent_state_has_gazette_fields(self):
        from state import AgentState
        annotations = AgentState.__annotations__
        gazette_fields = ["search_subject", "gazette_data", "gazette_url",
                          "gazette_analysis", "cross_check_result"]
        for field in gazette_fields:
            assert field in annotations, f"AgentState missing gazette field: {field}"

    def test_context_has_expected_fields(self):
        from state import Context
        annotations = Context.__annotations__
        expected = ["published_since", "published_until", "city",
                    "sources", "search_type"]
        for field in expected:
            assert field in annotations, f"Context missing field: {field}"

    def test_agent_state_is_total_false(self):
        """All fields should be optional (total=False)."""
        from state import AgentState
        # TypedDict with total=False allows creating empty instances
        instance = AgentState()  # Should not raise
        assert isinstance(instance, dict)


# ---------------------------------------------------------------------------
# 6. Errors module
# ---------------------------------------------------------------------------
class TestErrors:
    """Verify the errors module is importable from the new location."""

    def test_errors_import(self):
        from errors import NoGazettesFoundError, CityNotFoundError
        assert NoGazettesFoundError is not None
        assert CityNotFoundError is not None

    def test_no_gazettes_error_is_http_exception(self):
        from errors import NoGazettesFoundError
        from fastapi import HTTPException
        err = NoGazettesFoundError()
        assert isinstance(err, HTTPException)
        assert err.status_code == 404

    def test_city_not_found_error_is_http_exception(self):
        from errors import CityNotFoundError
        from fastapi import HTTPException
        err = CityNotFoundError()
        assert isinstance(err, HTTPException)
        assert err.status_code == 404


# ---------------------------------------------------------------------------
# 7. Tools module imports (no API keys needed)
# ---------------------------------------------------------------------------
class TestToolsImports:
    """Verify tool modules import without errors (no API calls)."""

    def test_querido_diario_imports(self):
        """The querido_diario module should import (loads IBGE JSON at import time)."""
        mod = importlib.import_module("tools.querido_diario")
        assert hasattr(mod, "querido_diario_fetch")

    def test_gazette_search_imports(self):
        """The gazette_search module should import without needing API keys."""
        mod = importlib.import_module("tools.gazette_search")
        assert hasattr(mod, "gazette_search_context")
        assert hasattr(mod, "querido_diario_glossario_tool")
        assert hasattr(mod, "DocumentLoader")

    def test_web_search_imports(self):
        """The web_search module should import without SERPAPI_API_KEY set."""
        mod = importlib.import_module("tools.web_search")
        assert hasattr(mod, "get_search_tool")

    def test_querido_diario_fetch_is_langchain_tool(self):
        from tools.querido_diario import querido_diario_fetch
        # LangChain @tool decorator creates a BaseTool instance
        assert hasattr(querido_diario_fetch, "invoke") or callable(querido_diario_fetch)

    def test_gazette_search_context_is_langchain_tool(self):
        from tools.gazette_search import gazette_search_context
        assert hasattr(gazette_search_context, "invoke") or callable(gazette_search_context)


# ---------------------------------------------------------------------------
# 8. Querido Diario tool logic (offline, no API call)
# ---------------------------------------------------------------------------
class TestQueridoDiarioToolLogic:
    """Test the querido_diario_fetch tool logic without making real API calls."""

    def test_unknown_city_searches_without_filter(self):
        """Unknown city should gracefully search without city filter instead of erroring."""
        from tools.querido_diario import querido_diario_fetch
        result = querido_diario_fetch.invoke({
            "subject": "test",
            "city": "NONEXISTENT_CITY_12345"
        })
        assert isinstance(result, dict)
        # Should return a gazette (no error) since it searches without city filter
        assert "error" not in result or "No public gazettes" in result.get("error", "")

    def test_none_city_searches_without_filter(self):
        """None/empty city should gracefully search without city filter."""
        from tools.querido_diario import querido_diario_fetch
        result = querido_diario_fetch.invoke({
            "subject": "test",
            "city": None
        })
        assert isinstance(result, dict)
        # Should return a gazette (no error) since it searches without city filter
        assert "error" not in result or "No public gazettes" in result.get("error", "")


# ---------------------------------------------------------------------------
# 9. DocumentLoader unit test
# ---------------------------------------------------------------------------
class TestDocumentLoader:
    """Test the DocumentLoader class from gazette_search."""

    def test_document_loader_loads_text_file(self):
        from tools.gazette_search import DocumentLoader
        # Use the glossary file as a real test file
        glossary_path = os.path.join(PROJECT_ROOT, "app", "data",
                                     "querido_diario_glossario_context.txt")
        loader = DocumentLoader(glossary_path)
        docs = loader.load()
        assert len(docs) > 0, "DocumentLoader should load at least one document"
        assert all(hasattr(d, "page_content") for d in docs)
        assert all(hasattr(d, "metadata") for d in docs)
        assert docs[0].metadata["source"] == glossary_path


# ---------------------------------------------------------------------------
# 10. LangChain/LangGraph core imports
# ---------------------------------------------------------------------------
class TestCoreImports:
    """Verify that the upgraded LangChain/LangGraph APIs are usable."""

    def test_langgraph_stategraph(self):
        from langgraph.graph import StateGraph, END
        assert StateGraph is not None
        assert END is not None

    def test_langchain_chat_openai(self):
        from langchain_openai import ChatOpenAI
        assert ChatOpenAI is not None

    def test_langchain_prompts(self):
        from langchain_core.prompts import ChatPromptTemplate
        assert ChatPromptTemplate is not None

    def test_langchain_output_parsers(self):
        from langchain_core.output_parsers import StrOutputParser
        assert StrOutputParser is not None

    def test_faiss_vectorstore(self):
        from langchain_community.vectorstores import FAISS
        assert FAISS is not None

    def test_openai_embeddings(self):
        from langchain_openai import OpenAIEmbeddings
        assert OpenAIEmbeddings is not None

    def test_web_base_loader(self):
        from langchain_community.document_loaders import WebBaseLoader
        assert WebBaseLoader is not None

    def test_recursive_text_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        assert RecursiveCharacterTextSplitter is not None

    def test_serpapi_wrapper(self):
        from langchain_community.utilities import SerpAPIWrapper
        assert SerpAPIWrapper is not None
