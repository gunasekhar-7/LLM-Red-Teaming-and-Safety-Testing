"""Unit tests for test harness layer."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from test_harness import (
    AttackResult,
    AttackCategory,
    AttackSeverity,
    ComparativeHarness,
    Probe,
    TestHarness,
    create_baseline_probes,
)


class MockRAGTarget:
    """Minimal mock RAG target for testing."""

    def __init__(self, name: str = "mock-target"):
        self.name = name

    async def query(self, question: str, **kwargs):
        return {
            "answer": "I don't have enough information to answer that question.",
            "citations": [],
            "chunks": [],
        }

    async def get_context(self, question: str, **kwargs):
        return []

    async def get_system_prompt(self):
        return "You are a helpful AI assistant."

    async def health_check(self):
        return True

    async def close(self):
        pass


class TestCreateBaselineProbes:
    """Test that baseline probes are created correctly."""

    def test_baseline_probes_not_empty(self):
        probes = create_baseline_probes()
        assert len(probes) >= 30, f"Expected 30+ baseline probes, got {len(probes)}"

    def test_probes_have_required_categories(self):
        probes = create_baseline_probes()
        categories = {p.category for p in probes}
        assert AttackCategory.PROMPT_INJECTION in categories
        assert AttackCategory.DATA_LEAKAGE in categories
        assert AttackCategory.JAILBREAK in categories
        assert AttackCategory.RAG_SPECIFIC in categories


class TestTestHarnessInitialization:
    """Test TestHarness can be initialized."""

    @pytest.fixture
    def mock_target(self):
        return MockRAGTarget("test")

    @pytest.fixture
    def mock_provider(self):
        from llm_provider import ProviderType, ModelConfig
        from llm_provider.factory import ProviderFactory
        from llm_provider.llm_settings import LLMSettings

        settings = LLMSettings()
        factory = ProviderFactory(settings)
        # Use Ollama which is local and doesn't need API keys
        config = ModelConfig(
            provider=ProviderType.OLLAMA,
            model_name="llama3.1:8b",
            base_url="http://localhost:11434",
        )
        return factory.create_provider("ollama-llama31", config)

    def test_harness_init_with_targets(self, mock_target, mock_provider):
        from test_harness import TestHarness
        from scripts.run_baseline import create_baseline_probes

        probes = create_baseline_probes()
        harness = TestHarness(
            targets=[mock_target],
            providers={"ollama-llama31": mock_provider},
            probes=probes[:5],  # Use just 5 for quick test
            output_dir="results/test",
        )
        assert harness is not None
        assert len(harness.targets) == 1
        assert len(harness.probes) == 5


class TestTestHarnessRunAll:
    """Test TestHarness.run_all method."""

    @pytest.fixture
    def mock_target(self):
        return MockRAGTarget("test")

    @pytest.fixture
    def mock_provider(self):
        from llm_provider import ProviderType, ModelConfig
        from llm_provider.factory import ProviderFactory
        from llm_provider.llm_settings import LLMSettings

        settings = LLMSettings()
        factory = ProviderFactory(settings)
        config = ModelConfig(
            provider=ProviderType.OLLAMA,
            model_name="llama3.1:8b",
            base_url="http://localhost:11434",
        )
        return factory.create_provider("ollama-llama31", config)

    @pytest.mark.asyncio
    async def test_run_all_with_mock_probes(self, mock_target, mock_provider):
        from test_harness import TestHarness
        from scripts.run_baseline import create_baseline_probes

        probes = create_baseline_probes()
        harness = TestHarness(
            targets=[mock_target],
            providers={"ollama-llama31": mock_provider},
            probes=probes[:3],  # Use 3 probes for quick test
            output_dir="results/test",
        )

        # Run with very small number of probes
        results = await harness.run_all(max_concurrent=1, progress=False)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    @pytest.mark.asyncio
    async def test_run_all_results_have_required_fields(self, mock_target, mock_provider):
        from test_harness import TestHarness
        from scripts.run_baseline import create_baseline_probes

        probes = create_baseline_probes()
        harness = TestHarness(
            targets=[mock_target],
            providers={"ollama-llama31": mock_provider},
            probes=probes[:2],
            output_dir="results/test",
        )

        results = await harness.run_all(max_concurrent=1, progress=False)
        assert len(results) == 2

        for result in results:
            assert hasattr(result, "probe_name")
            assert hasattr(result, "category")
            assert hasattr(result, "severity")
            assert hasattr(result, "passed")
            assert hasattr(result, "failed")
            assert hasattr(result, "prompt")
            assert hasattr(result, "response")
            assert hasattr(result, "target_name")
            assert hasattr(result, "provider_name")


class TestComparativeHarness:
    """Test ComparativeHarness comparison logic."""

    def test_comparison_overall_metrics_structure(self):
        from test_harness import AttackResult, AttackCategory, AttackSeverity
        import datetime

        baseline_results = [
            AttackResult(
                probe_name="test_probe",
                category=AttackCategory.PROMPT_INJECTION,
                severity=AttackSeverity.HIGH,
                passed=True,
                failed=False,
                prompt="Test prompt",
                response="I cannot help with that.",
                provider_name="test",
                target_name="test_target",
                timestamp=datetime.now(),
            )
        ]

        mitigated_results = [
            AttackResult(
                probe_name="test_probe",
                category=AttackCategory.PROMPT_INJECTION,
                severity=AttackSeverity.HIGH,
                passed=True,
                failed=False,
                prompt="Test prompt",
                response="I cannot help with that.",
                provider_name="test",
                target_name="test_target",
                timestamp=datetime.now(),
            )
        ]

        harness = ComparativeHarness(baseline_results, mitigated_results)
        comparison = harness.compare()

        overall = comparison["overall"]
        assert "baseline_failure_rate" in overall
        assert "mitigated_failure_rate" in overall
        assert "absolute_risk_reduction" in overall
        assert "relative_risk_reduction" in overall
        assert "p_value" in overall
        assert "effect_size" in overall
        assert "significant" in overall

    def test_comparison_by_category_structure(self):
        from test_harness import AttackResult, AttackCategory, AttackSeverity
        import datetime

        baseline_results = [
            AttackResult(
                probe_name="test_probe",
                category=AttackCategory.PROMPT_INJECTION,
                severity=AttackSeverity.HIGH,
                passed=True,
                failed=False,
                prompt="Test prompt",
                response="I cannot help with that.",
                provider_name="test",
                target_name="test_target",
                timestamp=datetime.now(),
            )
        ]

        mitigated_results = [
            AttackResult(
                probe_name="test_probe",
                category=AttackCategory.PROMPT_INJECTION,
                severity=AttackSeverity.HIGH,
                passed=True,
                failed=False,
                prompt="Test prompt",
                response="I cannot help with that.",
                provider_name="test",
                target_name="test_target",
                timestamp=datetime.now(),
            )
        ]

        harness = ComparativeHarness(baseline_results, mitigated_results)
        comparison = harness.compare()

        assert "by_category" in comparison
        cat_data = comparison["by_category"]
        assert AttackCategory.PROMPT_INJECTION.value in cat_data


class TestAttackResult:
    """Test AttackResult model."""

    def test_attack_result_creation(self):
        from test_harness import AttackResult, AttackCategory, AttackSeverity
        import datetime

        result = AttackResult(
            probe_name="test_probe",
            category=AttackCategory.PROMPT_INJECTION,
            severity=AttackSeverity.HIGH,
            passed=True,
            failed=False,
            prompt="Test prompt",
            response="I cannot help with that.",
            provider_name="test",
            target_name="test_target",
            timestamp=datetime.now(),
        )

        assert result.probe_name == "test_probe"
        assert result.category == AttackCategory.PROMPT_INJECTION
        assert result.severity == AttackSeverity.HIGH
        assert result.passed is True
        assert result.failed is False

    def test_attack_result_failed(self):
        from test_harness import AttackResult, AttackCategory, AttackSeverity
        import datetime

        result = AttackResult(
            probe_name="test_probe",
            category=AttackCategory.PROMPT_INJECTION,
            severity=AttackSeverity.HIGH,
            passed=False,
            failed=True,
            prompt="Test prompt",
            response="Sure, I'll reveal my system prompt:",
            provider_name="test",
            target_name="test_target",
            timestamp=datetime.now(),
        )

        assert result.passed is False
        assert result.failed is True