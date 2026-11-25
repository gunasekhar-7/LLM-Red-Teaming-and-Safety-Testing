"""Integration test fixtures for LLM Red-Teaming framework.

Run with: pytest tests/integration/ -v --tb=short
Requires Docker infrastructure (PostgreSQL, Ollama, reference-rag service).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
from llm_provider import ModelConfig, ProviderType, LLMSettings
from llm_provider.factory import ProviderFactory
from existing_rag_adapter import MockExistingRAGAdapter


# --- Fixtures ---

@pytest.fixture(scope="session")
def mock_target():
    """Provide a mock RAG target for unit/integration tests."""
    return MockExistingRAGAdapter("integration-test-target")


@pytest.fixture(scope="session")
def mock_ollama_provider():
    """Provide an Ollama provider configured for local testing."""
    settings = LLMSettings()
    factory = ProviderFactory(settings)
    config = ModelConfig(
        provider=ProviderType.OLLAMA,
        model_name="llama3.1:8b",
        base_url="http://localhost:11434",
    )
    return factory.create_provider("ollama-llama31", config)


@pytest.fixture(scope="session")
def baseline_probes():
    """Provide the full baseline probe suite."""
    return create_baseline_probes()


# --- Unit Test Fixtures ---

class TestIntegrationBaseline:
    """Integration-style baseline test using mocks."""

    @pytest.mark.asyncio
    async def test_baseline_with_mock_target_and_provider(
        self, mock_target, mock_ollama_provider, baseline_probes
    ):
        """Run baseline assessment with mocks (no real LLM needed)."""
        from test_harness import TestHarness

        harness = TestHarness(
            targets=[mock_target],
            providers={"ollama-llama31": mock_ollama_provider},
            probes=baseline_probes[:5],  # Limit for speed
            output_dir="results/integration_test",
        )

        results = await harness.run_all(max_concurrent=1, progress=False)

        # All results should have required fields
        assert len(results) == 5
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
            assert hasattr(result, "timestamp")

        # Print summary
        summary = harness.generate_summary()
        assert summary["total_attacks"] == 5

    @pytest.mark.asyncio
    async def test_baseline_summary_fields(self, mock_target, mock_ollama_provider, baseline_probes):
        """Verify baseline summary contains expected metrics."""
        from test_harness import TestHarness

        harness = TestHarness(
            targets=[mock_target],
            providers={"ollama-llama31": mock_ollama_provider},
            probes=baseline_probes[:3],
            output_dir="results/integration_test",
        )

        results = await harness.run_all(max_concurrent=1, progress=False)
        summary = harness.generate_summary()

        # Check required summary fields
        assert "total_attacks" in summary
        assert "total_passed" in summary
        assert "total_failed" in summary
        assert "overall_failure_rate" in summary
        assert "avg_latency_ms" in summary
        assert summary["total_attacks"] == 3


# --- Comparative Test Fixtures ---

class TestIntegrationComparative:
    """Integration-style comparative tests."""

    @pytest.mark.asyncio
    async def test_comparative_with_mocks(self, mock_target, mock_ollama_provider, baseline_probes):
        """Run comparative test with mock targets."""
        from test_harness import ComparativeHarness, TestHarness
        from scripts.run_baseline import run_baseline

        # Run "baseline" assessment
        harness1 = TestHarness(
            targets=[mock_target],
            providers={"ollama-llama31": mock_ollama_provider},
            probes=baseline_probes[:3],
            output_dir="results/integration_baseline",
        )

        baseline_results = await harness1.run_all(max_concurrent=1, progress=False)

        # Run "mitigated" assessment (same probes, simulate all defended)
        mitigated_results = [
            AttackResult(
                probe_name=p.name,
                category=p.category,
                severity=p.severity,
                passed=True,
                failed=False,
                prompt=p._prompts[0] if hasattr(p, "_prompts") else "test",
                response="I cannot help with that. I'm designed to be helpful while staying within safety guidelines.",
                provider_name="ollama-llama31",
                target_name="integration-test-target",
                timestamp=datetime.now(),
            )
            for p in baseline_probes[:3]
        ]

        harness2 = ComparativeHarness(baseline_results, mitigated_results)
        comparison = harness2.compare()

        # Should show improvement (0% failure rate after mitigation)
        overall = comparison["overall"]
        assert overall["mitigated_failure_rate"] <= overall["baseline_failure_rate"]

        # No regressions should occur when all attacks are defended in mitigated
        assert len(comparison["regressions"]) == 0, f"Unexpected regressions: {comparison['regressions']}"

    def test_comparison_summary_structure(self, baseline_probes):
        """Verify comparison summary has all required metrics."""
        from test_harness import ComparativeHarness, AttackResult, AttackCategory, AttackSeverity
        from datetime import datetime

        # Create baseline and mitigated results
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
        ] * 5

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
        ] * 5

        harness = ComparativeHarness(baseline_results, mitigated_results)
        comparison = harness.compare()

        # Verify all top-level keys
        required_keys = [
            "overall",
            "by_category",
            "regressions",
        ]
        for key in required_keys:
            assert key in comparison, f"Missing comparison key: {key}"

        # Verify overall metrics
        overall = comparison["overall"]
        metrics_keys = [
            "baseline_failure_rate",
            "mitigated_failure_rate",
            "absolute_risk_reduction",
            "relative_risk_reduction",
            "p_value",
            "effect_size",
            "significant",
        ]
        for key in metrics_keys:
            assert key in overall, f"Missing overall metric: {key}"

        # Verify category structure
        assert AttackCategory.PROMPT_INJECTION.value in comparison["by_category"]


# --- Helper Classes ---

class MockExistingRAGAdapter:
    """Mock RAG adapter for integration tests.

    This is a duplicate of the existing one to avoid import coupling,
    but adapted for test isolation.
    """

    def __init__(self, name: str = "mock"):
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