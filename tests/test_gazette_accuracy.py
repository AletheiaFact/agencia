"""
Gazette Search Accuracy Tests
==============================
Tests for the adaptive gazette search pipeline.

Three levels:
1. Tool-level tests (offline, no API keys needed) - use cached fixtures
2. Integration tests (need OPENAI_API_KEY) - test individual LLM nodes
3. E2E golden tests (need OPENAI_API_KEY + API access) - full pipeline

Run offline tests:   python -m pytest tests/test_gazette_accuracy.py -v -k "not integration and not e2e"
Run integration:     python -m pytest tests/test_gazette_accuracy.py -v -m integration
Run E2E:             python -m pytest tests/test_gazette_accuracy.py -v -m e2e
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# Ensure app/ is on the path
APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
API_RESPONSES_DIR = os.path.join(FIXTURES_DIR, "gazette_api_responses")
GAZETTE_TEXTS_DIR = os.path.join(FIXTURES_DIR, "gazette_texts")


def _load_fixture(filename: str) -> dict:
    """Load a JSON fixture file."""
    path = os.path.join(API_RESPONSES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text_fixture(filename: str) -> str:
    """Load a text fixture file."""
    path = os.path.join(GAZETTE_TEXTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. Tool-level tests (offline, no API keys needed)
# ---------------------------------------------------------------------------
class TestQueridoDiarioSearchTool:
    """Test the new querido_diario_search tool logic."""

    def test_import(self):
        from tools.querido_diario_search import querido_diario_search
        assert hasattr(querido_diario_search, "invoke")

    def test_resolve_territory_ids_known_city(self):
        from tools.querido_diario_search import _resolve_territory_ids
        ids = _resolve_territory_ids("Porto Alegre")
        assert len(ids) > 0
        assert "4314902" in ids

    def test_resolve_territory_ids_unknown_city(self):
        from tools.querido_diario_search import _resolve_territory_ids
        ids = _resolve_territory_ids("NONEXISTENT_CITY")
        assert ids == []

    def test_resolve_territory_ids_empty(self):
        from tools.querido_diario_search import _resolve_territory_ids
        assert _resolve_territory_ids("") == []
        assert _resolve_territory_ids("   ") == []

    @patch("tools.querido_diario_search.requests.get")
    def test_multi_query_merges_results(self, mock_get):
        """Multiple queries should merge and deduplicate by txt_url."""
        from tools.querido_diario_search import querido_diario_search

        fixture = _load_fixture("porto_alegre_flood.json")

        # First query returns all 5 gazettes
        # Second query returns a subset (3 of the same) plus will be deduped
        response_1 = MagicMock()
        response_1.json.return_value = fixture
        response_1.raise_for_status = MagicMock()

        # Create a subset for second query
        subset = {"total_gazettes": 3, "gazettes": fixture["gazettes"][:3]}
        response_2 = MagicMock()
        response_2.json.return_value = subset
        response_2.raise_for_status = MagicMock()

        mock_get.side_effect = [response_1, response_2]

        result = querido_diario_search.invoke({
            "queries": ["query1", "query2"],
            "city": "Porto Alegre",
        })

        assert result["total_unique"] == 5  # Deduped: 5 unique from first + 0 new from second
        assert len(result["query_stats"]) == 2
        assert result["query_stats"][0]["total_gazettes"] == 20
        assert mock_get.call_count == 2

    @patch("tools.querido_diario_search.requests.get")
    def test_empty_results(self, mock_get):
        """Empty API response should return empty candidates."""
        from tools.querido_diario_search import querido_diario_search

        fixture = _load_fixture("empty_result.json")
        mock_response = MagicMock()
        mock_response.json.return_value = fixture
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = querido_diario_search.invoke({
            "queries": ["alien invasion"],
            "city": "São Paulo",
        })

        assert result["total_unique"] == 0
        assert result["gazettes"] == []

    @patch("tools.querido_diario_search.requests.get")
    def test_query_params_include_excerpts(self, mock_get):
        """API call should include excerpt_size and number_of_excerpts params."""
        from tools.querido_diario_search import querido_diario_search

        mock_response = MagicMock()
        mock_response.json.return_value = {"total_gazettes": 0, "gazettes": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        querido_diario_search.invoke({
            "queries": ["test"],
            "size": 30,
            "excerpt_size": 500,
            "number_of_excerpts": 3,
        })

        call_url = mock_get.call_args[0][0]
        assert "excerpt_size=500" in call_url
        assert "number_of_excerpts=3" in call_url
        assert "size=30" in call_url

    @patch("tools.querido_diario_search.requests.get")
    def test_http_error_handled_gracefully(self, mock_get):
        """HTTP errors on one query should not crash the whole search."""
        from tools.querido_diario_search import querido_diario_search
        import requests

        # First query fails, second succeeds
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("connection failed"),
            MagicMock(
                json=MagicMock(return_value={"total_gazettes": 1, "gazettes": [
                    {"txt_url": "http://example.com/1.txt", "excerpts": ["test"]}
                ]}),
                raise_for_status=MagicMock(),
            ),
        ]

        result = querido_diario_search.invoke({
            "queries": ["bad_query", "good_query"],
        })

        assert result["total_unique"] == 1
        assert "error" in result["query_stats"][0]
        assert result["query_stats"][1]["total_gazettes"] == 1


class TestGazetteDeepSearchTool:
    """Test the new gazette_deep_search tool logic."""

    def test_import(self):
        from tools.gazette_deep_search import gazette_deep_search
        assert hasattr(gazette_deep_search, "invoke")

    @patch("tools.gazette_deep_search._download_gazette_text")
    @patch("tools.gazette_deep_search.OpenAIEmbeddings")
    @patch("tools.gazette_deep_search.FAISS")
    def test_multi_doc_builds_unified_index(self, mock_faiss, mock_embed, mock_download):
        """Multiple URLs should be combined into a single FAISS index."""
        from tools.gazette_deep_search import gazette_deep_search

        sample_text = _load_text_fixture("porto_alegre_flood_sample.txt")
        mock_download.return_value = sample_text

        # Mock FAISS chain
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            MagicMock(page_content="test chunk 1", metadata={"source": "url1"}),
            MagicMock(page_content="test chunk 2", metadata={"source": "url2"}),
        ]
        mock_vectorstore = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vectorstore

        result = gazette_deep_search.invoke({
            "claim": "test claim",
            "urls": ["http://example.com/1.txt", "http://example.com/2.txt"],
        })

        assert "test chunk 1" in result
        assert "test chunk 2" in result
        assert mock_download.call_count == 2

    @patch("tools.gazette_deep_search._download_gazette_text")
    def test_all_downloads_fail_returns_error(self, mock_download):
        """If all downloads fail, return error string."""
        from tools.gazette_deep_search import gazette_deep_search

        mock_download.return_value = None

        result = gazette_deep_search.invoke({
            "claim": "test claim",
            "urls": ["http://bad1.txt", "http://bad2.txt"],
        })

        assert result.startswith("Error:")


class TestFixturesIntegrity:
    """Verify test fixtures are valid and accessible."""

    def test_porto_alegre_fixture_has_gazettes(self):
        data = _load_fixture("porto_alegre_flood.json")
        assert data["total_gazettes"] == 20
        assert len(data["gazettes"]) == 5
        assert data["gazettes"][0]["territory_name"] == "Porto Alegre"

    def test_rio_fixture_has_gazettes(self):
        data = _load_fixture("rio_medicamentos.json")
        assert data["total_gazettes"] == 239
        assert len(data["gazettes"]) >= 1

    def test_bh_fixture_has_gazettes(self):
        data = _load_fixture("bh_guarda_municipal.json")
        assert data["total_gazettes"] == 77
        assert len(data["gazettes"]) >= 1

    def test_curitiba_fixture_has_gazettes(self):
        data = _load_fixture("curitiba_enfermeiros.json")
        assert data["total_gazettes"] == 129
        assert len(data["gazettes"]) >= 1

    def test_empty_fixture(self):
        data = _load_fixture("empty_result.json")
        assert data["total_gazettes"] == 0
        assert data["gazettes"] == []

    def test_gazette_text_fixture_not_empty(self):
        text = _load_text_fixture("porto_alegre_flood_sample.txt")
        assert len(text) > 100

    def test_all_fixtures_have_excerpts(self):
        """Non-empty fixtures should have excerpts for scoring."""
        for name in ["porto_alegre_flood.json", "rio_medicamentos.json",
                      "bh_guarda_municipal.json", "curitiba_enfermeiros.json"]:
            data = _load_fixture(name)
            for gazette in data["gazettes"]:
                assert "excerpts" in gazette, f"{name}: gazette missing excerpts"
                assert len(gazette["excerpts"]) > 0, f"{name}: gazette has empty excerpts"


class TestSubgraphStructure:
    """Test the subgraph can be built (no LLM calls)."""

    def test_build_subgraph_compiles(self):
        """The gazette subgraph should compile without errors."""
        from nodes.gazette.subgraph import build_gazette_subgraph
        graph = build_gazette_subgraph()
        assert graph is not None

    def test_subgraph_has_expected_nodes(self):
        """Verify all expected nodes are in the compiled graph."""
        from nodes.gazette.subgraph import build_gazette_subgraph
        graph = build_gazette_subgraph()
        # LangGraph compiled graph stores node names
        node_names = set(graph.nodes.keys())
        expected = {
            "plan_search", "fetch_and_score", "evaluate_evidence",
            "refine_queries", "download_and_analyze", "cross_check",
            "gazette_report",
        }
        # __start__ and __end__ are added by LangGraph
        for name in expected:
            assert name in node_names, f"Missing node: {name}"


class TestEvidenceEvaluatorLogic:
    """Test the evidence evaluator's non-LLM logic paths."""

    def test_max_iteration_forces_sufficient(self):
        from nodes.gazette.evidence_evaluator import evaluate_evidence
        state = {
            "claim": "test claim",
            "search_iteration": 2,
            "gazette_candidates": [],
            "evidence_summary": "",
            "search_strategies": [],
        }
        result = evaluate_evidence(state)
        assert result["evidence_sufficient"] is True

    def test_no_candidates_is_insufficient(self):
        from nodes.gazette.evidence_evaluator import evaluate_evidence
        state = {
            "claim": "test claim",
            "search_iteration": 0,
            "gazette_candidates": [],
            "evidence_summary": "",
            "search_strategies": [],
        }
        result = evaluate_evidence(state)
        assert result["evidence_sufficient"] is False

    def test_high_score_is_sufficient(self):
        from nodes.gazette.evidence_evaluator import evaluate_evidence
        state = {
            "claim": "test claim",
            "search_iteration": 0,
            "gazette_candidates": [{"_relevance_score": 8}],
            "evidence_summary": "strong evidence here",
            "search_strategies": ["query1"],
        }
        result = evaluate_evidence(state)
        assert result["evidence_sufficient"] is True


class TestStateSchema:
    """Verify the updated state schema has new fields."""

    def test_new_gazette_fields_exist(self):
        from state import AgentState
        annotations = AgentState.__annotations__
        new_fields = [
            "search_strategies", "gazette_candidates", "evidence_sufficient",
            "search_iteration", "selected_gazettes", "evidence_summary",
        ]
        for field in new_fields:
            assert field in annotations, f"AgentState missing new field: {field}"

    def test_backward_compatible_fields(self):
        """Core fields should still exist."""
        from state import AgentState
        annotations = AgentState.__annotations__
        core = ["claim", "context", "messages", "questions", "search_type", "language"]
        for field in core:
            assert field in annotations, f"AgentState missing core field: {field}"


# ---------------------------------------------------------------------------
# 2. Integration tests (need OPENAI_API_KEY)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestSearchPlannerIntegration:
    """Test the search planner node with real LLM calls."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_generates_multiple_strategies(self):
        from nodes.gazette.search_planner import plan_search
        state = {
            "claim": "A Prefeitura de Porto Alegre firmou contratos emergenciais com empresas privadas para limpeza urbana.",
            "context": {
                "city": "Porto Alegre",
                "published_since": "2024-05-01",
                "published_until": "2024-07-31",
            },
            "language": "pt",
        }
        result = plan_search(state)
        strategies = result["search_strategies"]
        assert len(strategies) >= 2, f"Expected 2+ strategies, got {len(strategies)}"
        assert all(isinstance(s, str) and len(s) > 0 for s in strategies)
        assert result["search_iteration"] == 0
        assert result["evidence_sufficient"] is False


@pytest.mark.integration
class TestCrossCheckerIntegration:
    """Test the upgraded cross-checker with real LLM."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_classifies_with_strong_evidence(self):
        from nodes.gazette.cross_checker import cross_check
        state = {
            "claim": "A Prefeitura de Porto Alegre firmou contratos emergenciais com empresas privadas.",
            "gazette_analysis": "Found CONTRATO EMERGENCIAL 002/2024 between DMLU and LOCAR VEICULOS E EQUIPAMENTOS LTDA. Contract 004/2024 with FG SOLUCOES AMBIENTAIS LTDA.",
            "evidence_summary": "Multiple emergency contracts found in Porto Alegre gazettes from May-June 2024.",
            "selected_gazettes": [
                {"territory_name": "Porto Alegre", "date": "2024-05-17", "txt_url": "http://example.com", "relevance_score": 9},
            ],
        }
        result = cross_check(state)
        assert "cross_check_result" in result
        assert len(result["cross_check_result"]) > 100
        # Should classify as Trustworthy or similar positive classification
        text = result["cross_check_result"].lower()
        assert any(word in text for word in ["trustworthy", "confiável", "verdadeiro", "reliable"])


# ---------------------------------------------------------------------------
# 3. E2E Golden Tests (need OPENAI_API_KEY + API access)
# ---------------------------------------------------------------------------
@pytest.mark.e2e
class TestGoldenPortoAlegre:
    """E2E: Porto Alegre emergency flood contracts → Trustworthy."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_full_pipeline(self):
        from nodes.gazette.subgraph import build_gazette_subgraph

        graph = build_gazette_subgraph()
        state = {
            "claim": "A Prefeitura de Porto Alegre firmou contratos emergenciais com empresas privadas para limpeza urbana apos as enchentes de maio de 2024.",
            "context": {
                "city": "Porto Alegre",
                "published_since": "2024-05-01",
                "published_until": "2024-07-31",
            },
            "questions": ["Quais contratos emergenciais foram firmados?", "Com quais empresas?"],
            "language": "pt",
        }
        result = graph.invoke(state)

        assert "messages" in result
        assert len(result["messages"]) > 100
        # Should find evidence and classify positively
        msg = result["messages"].lower()
        assert "trustworthy" in msg or "confiável" in msg or "verdadeiro" in msg


@pytest.mark.e2e
class TestGoldenNegative:
    """E2E: Absurd claim → Unverifiable."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_full_pipeline_no_results(self):
        from nodes.gazette.subgraph import build_gazette_subgraph

        graph = build_gazette_subgraph()
        state = {
            "claim": "Houve invasao alienigena registrada no diario oficial de Sao Paulo em 2024.",
            "context": {
                "city": "São Paulo",
                "published_since": "2024-01-01",
                "published_until": "2024-12-31",
            },
            "questions": [],
            "language": "pt",
        }
        result = graph.invoke(state)

        assert "messages" in result
        msg = result["messages"].lower()
        assert "unverifiable" in msg or "inverificável" in msg or "não verificável" in msg
