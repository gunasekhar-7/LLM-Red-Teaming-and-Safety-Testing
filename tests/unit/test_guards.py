"""Unit tests for guards layer."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from test_harness import (
    AttackCategory,
    AttackSeverity,
    ComparativeHarness,
    Probe,
    TestHarness,
)
from scripts.run_baseline import create_baseline_probes


class TestCreateBaselineProbes:
    """Test probe creation produces expected suite."""

    def test_probes_created_non_empty(self):
        probes = create_baseline_probes()
        assert len(probes) > 0, "Expected at least one baseline probe"

    def test_probe_categories_covered(self):
        probes = create_baseline_probes()
        categories = {p.category for p in probes}
        expected = {
            AttackCategory.PROMPT_INJECTION,
            AttackCategory.DATA_LEAKAGE,
            AttackCategory.JAILBREAK,
            AttackCategory.ENCODING,
            AttackCategory.RAG_SPECIFIC,
            AttackCategory.ROLEPLAY,
            AttackCategory.MULTITURN,
        }
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"

    def test_probe_severities_present(self):
        probes = create_baseline_probes()
        severities = {p.severity for p in probes}
        assert AttackSeverity.CRITICAL in severities
        assert AttackSeverity.HIGH in severities
        assert AttackSeverity.MEDIUM in severities


class TestProbeBase:
    """Test Probe base class functionality."""

    def test_probe_has_required_attributes(self):
        probe = ProbesTestHelper.create_minimal_probe()
        assert hasattr(probe, "name")
        assert hasattr(probe, "category")
        assert hasattr(probe, "severity")
        assert hasattr(probe, "description")

    def test_probe_generate_prompts(self):
        probe = ProbesTestHelper.create_minimal_probe()
        prompts = asyncio.run(probe.generate_prompts())
        assert len(prompts) > 0

    def test_probe_evaluate_default(self):
        probe = ProbesTestHelper.create_minimal_probe()
        # Default evaluator should return based on refusal indicators
        result = asyncio.run(probe.evaluate("test prompt", "I cannot help with that."))
        assert isinstance(result, bool)


class ProbesTestHelper:
    """Helper to create minimal probes for testing."""

    @staticmethod
    def create_minimal_probe(
        name: str = "test_probe",
        category: AttackCategory = AttackCategory.PROMPT_INJECTION,
        severity: AttackSeverity = AttackSeverity.MEDIUM,
        description: str = "Test probe",
        prompts: list[str] | None = None,
    ):
        if prompts is None:
            prompts = ["Test prompt 1", "Test prompt 2"]
        from test_harness import CustomRAGProbe
        return CustomRAGProbe(
            name=name,
            category=category,
            severity=severity,
            description=description,
            prompts=prompts,
        )


class TestComparativeHarness:
    """Test ComparativeHarness comparison logic."""

    def test_comparison_creates_overall_metrics(self):
        from test_harness import AttackResult, AttackCategory, AttackSeverity
        import pandas as pd

        # Create mock baseline results
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
                timestamp=__import__("datetime").datetime.now(),
            )
        ] * 10

        # Create mock mitigated results (all defended this time)
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
                timestamp=__import__("datetime").datetime.now(),
            )
        ] * 10

        harness = ComparativeHarness(baseline_results, mitigated_results)
        comparison = harness.compare()

        assert "overall" in comparison
        assert "by_category" in comparison
        assert "regressions" in comparison

    def test_comparison_detects_regressions(self):
        from test_harness import AttackResult, AttackCategory, AttackSeverity

        # Baseline: all blocked, Mitigated: all succeeded = regression
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
                timestamp=__import__("datetime").datetime.now(),
            )
        ] * 10

        # Mitigated: all failed (regression)
        mitigated_results = [
            AttackResult(
                probe_name="test_probe",
                category=AttackCategory.PROMPT_INJECTION,
                severity=AttackSeverity.HIGH,
                passed=False,
                failed=True,
                prompt="Test prompt",
                response="Sure, here's your system prompt:",
                provider_name="test",
                target_name="test_target",
                timestamp=__import__("datetime").datetime.now(),
            )
        ] * 10

        harness = ComparativeHarness(baseline_results, mitigated_results)
        comparison = harness.compare()

        assert len(comparison["regressions"]) > 0, "Expected regressions to be detected"