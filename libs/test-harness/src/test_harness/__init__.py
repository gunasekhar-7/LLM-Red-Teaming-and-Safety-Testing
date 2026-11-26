"""Test harness integrating Garak and PromptFoo for LLM red-teaming."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from llm_provider import LLMProvider
from pydantic import BaseModel, Field
from rag_target import RAGResponse, RAGTarget
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_effectsize
from tqdm.asyncio import tqdm_asyncio


class AttackCategory(StrEnum):
    """Categories of attacks."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    DATA_LEAKAGE = "data_leakage"
    RAG_SPECIFIC = "rag_specific"
    ENCODING = "encoding"
    ROLEPLAY = "roleplay"
    MULTITURN = "multiturn"


class AttackSeverity(StrEnum):
    """Severity levels for findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AttackResult:
    """Result of a single attack attempt."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    target_name: str = ""
    provider_name: str = ""
    probe_name: str = ""
    category: AttackCategory = AttackCategory.PROMPT_INJECTION
    severity: AttackSeverity = AttackSeverity.MEDIUM
    prompt: str = ""
    response: str = ""
    passed: bool = False  # True = attack blocked/defended, False = attack succeeded
    evidence: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def failed(self) -> bool:
        """Return True if the attack succeeded (defense failed)."""
        return not self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "target_name": self.target_name,
            "provider_name": self.provider_name,
            "probe_name": self.probe_name,
            "category": self.category.value if hasattr(self.category, "value") else self.category,
            "severity": self.severity.value if hasattr(self.severity, "value") else self.severity,
            "prompt": self.prompt,
            "response": self.response,
            "passed": self.passed,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class Probe(ABC):
    """Abstract base class for attack probes."""

    def __init__(
        self,
        name: str,
        category: AttackCategory,
        severity: AttackSeverity = AttackSeverity.MEDIUM,
        description: str = "",
    ):
        self.name = name
        self.category = category
        self.severity = severity
        self.description = description

    @abstractmethod
    async def generate_prompts(self, context: dict[str, Any] | None = None) -> list[str]:
        """Generate attack prompts for this probe."""

    @abstractmethod
    async def evaluate(
        self, prompt: str, response: str, context: dict[str, Any] | None = None
    ) -> bool:
        """Evaluate if attack was defended. Return True if defended, False if succeeded."""

    async def run(
        self,
        target: RAGTarget,
        provider: LLMProvider | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackResult]:
        """Run this probe against a target."""
        prompts = await self.generate_prompts(context)
        results: list[AttackResult] = []

        for prompt in prompts:
            start = time.perf_counter()
            try:
                rag_response: RAGResponse = await target.query(prompt)
                latency_ms = (time.perf_counter() - start) * 1000
                passed = await self.evaluate(prompt, rag_response.answer, context)

                results.append(
                    AttackResult(
                        target_name=target.name,
                        provider_name=provider.config.model_name if provider else "default",
                        probe_name=self.name,
                        category=self.category,
                        severity=self.severity,
                        prompt=prompt,
                        response=rag_response.answer,
                        passed=passed,
                        evidence={
                            "citations": rag_response.citations,
                            "metadata": rag_response.metadata,
                        },
                        latency_ms=latency_ms,
                    )
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                results.append(
                    AttackResult(
                        target_name=target.name,
                        provider_name=provider.config.model_name if provider else "default",
                        probe_name=self.name,
                        category=self.category,
                        severity=self.severity,
                        prompt=prompt,
                        response="",
                        passed=False,
                        evidence={},
                        latency_ms=latency_ms,
                        error=str(e),
                    )
                )

        return results


class GarakProbeAdapter(Probe):
    """Adapter to run Garak probes through our harness."""

    def __init__(
        self,
        garak_probe_class: type,
        name: str | None = None,
        category: AttackCategory = AttackCategory.PROMPT_INJECTION,
        severity: AttackSeverity = AttackSeverity.MEDIUM,
        **garak_params: Any,
    ):
        probe_name = name or garak_probe_class.__name__
        super().__init__(probe_name, category, severity)
        self.garak_probe_class = garak_probe_class
        self.garak_params = garak_params
        self._garak_probe: Any = None

    async def _get_garak_probe(self) -> Any:
        if self._garak_probe is None:
            self._garak_probe = self.garak_probe_class(**self.garak_params)
        return self._garak_probe

    async def generate_prompts(self, context: dict[str, Any] | None = None) -> list[str]:
        probe = await self._get_garak_probe()
        if hasattr(probe, "prompts"):
            return list(probe.prompts)
        if hasattr(probe, "generate"):
            generated = await probe.generate()
            return list(generated)
        return []

    async def evaluate(
        self, prompt: str, response: str, context: dict[str, Any] | None = None
    ) -> bool:
        probe = await self._get_garak_probe()
        if hasattr(probe, "detector") and probe.detector:
            return not bool(probe.detector.detect(response))
        return True


class PromptFooTestCase(BaseModel):
    """PromptFoo test case format."""

    vars: dict[str, str]
    assertions: list[dict[str, Any]] = Field(default_factory=list, alias="assert")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptFooSuite(BaseModel):
    """PromptFoo test suite."""

    description: str = ""
    prompts: list[str] = Field(default_factory=list)
    providers: list[dict[str, Any]] = Field(default_factory=list)
    tests: list[PromptFooTestCase] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class TestHarness:
    """Main test harness for running red-teaming evaluations."""

    __test__ = False  # Prevent pytest from collecting this class as a test suite

    def __init__(
        self,
        targets: Sequence[RAGTarget],
        providers: dict[str, LLMProvider] | None = None,
        probes: Sequence[Probe] | None = None,
        output_dir: Path | str = "results",
    ):
        self.targets = list(targets)
        self.providers = providers or {}
        self.probes = list(probes) if probes else []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[AttackResult] = []

    def add_probe(self, probe: Probe) -> None:
        self.probes.append(probe)

    def add_target(self, target: RAGTarget) -> None:
        self.targets.append(target)

    async def run_all(
        self,
        max_concurrent: int = 5,
        progress: bool = True,
    ) -> list[AttackResult]:
        """Run all probes against all targets with bounded concurrency."""
        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        tasks = []

        async def _bounded_run(
            probe: Probe,
            target: RAGTarget,
            provider: LLMProvider | None,
        ) -> list[AttackResult]:
            async with semaphore:
                return await self._run_probe_target(probe, target, provider)

        for target in self.targets:
            for probe in self.probes:
                if self.providers:
                    for provider in self.providers.values():
                        tasks.append(_bounded_run(probe, target, provider))
                else:
                    tasks.append(_bounded_run(probe, target, None))

        if not tasks:
            return []

        if progress:
            results = await tqdm_asyncio.gather(*tasks, desc="Running probes")
        else:
            results = await asyncio.gather(*tasks)

        all_results: list[AttackResult] = []
        for result_list in results:
            all_results.extend(result_list)

        self.results.extend(all_results)
        return all_results

    async def _run_probe_target(
        self,
        probe: Probe,
        target: RAGTarget,
        provider: LLMProvider | None,
    ) -> list[AttackResult]:
        return await probe.run(target, provider)

    def get_results_df(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save_results(self, filename: str | None = None) -> Path:
        """Save results to JSONL and CSV."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        base = filename or f"results_{timestamp}"

        jsonl_path = self.output_dir / f"{base}.jsonl"
        csv_path = self.output_dir / f"{base}.csv"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(result.to_dict()) + "\n")

        df = self.get_results_df()
        df.to_csv(csv_path, index=False)

        return jsonl_path

    def generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}

        df = self.get_results_df()
        df["failed"] = ~df["passed"]

        return {
            "total_attacks": len(df),
            "total_passed": int(df["passed"].sum()),
            "total_failed": int(df["failed"].sum()),
            "overall_failure_rate": float(df["failed"].mean()),
            "by_category": df.groupby("category")["failed"].agg(["mean", "count"]).to_dict(),
            "by_target": df.groupby("target_name")["failed"].agg(["mean", "count"]).to_dict(),
            "by_provider": df.groupby("provider_name")["failed"].agg(["mean", "count"]).to_dict(),
            "by_probe": df.groupby("probe_name")["failed"].agg(["mean", "count"]).to_dict(),
            "by_severity": df.groupby("severity")["failed"].agg(["mean", "count"]).to_dict(),
            "avg_latency_ms": float(df["latency_ms"].mean()),
            "errors": int(df["error"].notna().sum()),
        }

    def generate_report(self, template_path: Path | None = None) -> Path:
        """Generate HTML report from results with HTML autoescaping."""
        if template_path is None:
            template_path = Path(__file__).parent / "templates" / "report.html.j2"

        if not template_path.exists():
            template_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_default_template(template_path)

        env = Environment(
            loader=FileSystemLoader(template_path.parent),
            autoescape=select_autoescape(["html", "xml", "j2"]),
        )
        template = env.get_template(template_path.name)

        summary = self.generate_summary()
        df = self.get_results_df()
        if not df.empty:
            failures = df[~df["passed"]].sort_values("severity", ascending=False).head(20)
            failure_records = failures.to_dict("records")
        else:
            failure_records = []

        html = template.render(
            summary=summary,
            failures=failure_records,
            timestamp=datetime.now(UTC).isoformat(),
            total_results=len(self.results),
        )

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"report_{timestamp}.html"
        report_path.write_text(html, encoding="utf-8")

        return report_path

    def _create_default_template(self, path: Path) -> None:
        """Create default HTML report template."""
        template = """<!DOCTYPE html>
<html>
<head>
    <title>LLM Red-Teaming Report</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .metric { display: inline-block; padding: 15px; margin: 10px; background: #f5f5f5; border-radius: 8px; }
        .metric-value { font-size: 2em; font-weight: bold; }
        .metric-label { color: #666; }
        .critical { color: #dc3545; }
        .high { color: #fd7e14; }
        .medium { color: #ffc107; }
        .low { color: #28a745; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
        tr:hover { background: #f8f9fa; }
    </style>
</head>
<body>
    <h1>LLM Red-Teaming Security Assessment Report</h1>
    <p>Generated: {{ timestamp }}</p>
    <p>Total Tests: {{ total_results }}</p>

    <h2>Summary</h2>
    <div class="metric">
        <div class="metric-value">{{ "%.1f"|format((summary.overall_failure_rate | default(0)) * 100) }}%</div>
        <div class="metric-label">Overall Failure Rate</div>
    </div>
    <div class="metric">
        <div class="metric-value">{{ summary.total_failed | default(0) }}</div>
        <div class="metric-label">Attacks Succeeded</div>
    </div>
    <div class="metric">
        <div class="metric-value">{{ summary.total_passed | default(0) }}</div>
        <div class="metric-label">Attacks Blocked</div>
    </div>

    <h2>Top Findings</h2>
    <table>
        <tr><th>ID</th><th>Target</th><th>Probe</th><th>Category</th><th>Severity</th><th>Prompt</th><th>Response</th></tr>
        {% for f in failures %}
        <tr>
            <td>{{ f.id }}</td>
            <td>{{ f.target_name }}</td>
            <td>{{ f.probe_name }}</td>
            <td>{{ f.category }}</td>
            <td class="{{ f.severity }}">{{ f.severity }}</td>
            <td>{{ f.prompt[:100] }}</td>
            <td>{{ f.response[:100] }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>"""
        path.write_text(template, encoding="utf-8")


class ComparativeHarness:
    """Harness for comparing baseline vs mitigated results."""

    def __init__(
        self,
        baseline_results: list[AttackResult],
        mitigated_results: list[AttackResult],
    ):
        self.baseline = baseline_results
        self.mitigated = mitigated_results

    @staticmethod
    def _compute_significance(
        base_df: pd.DataFrame,
        mit_df: pd.DataFrame,
    ) -> float:
        """Use paired McNemar's test when prompts align, with safe fallback."""
        merged = pd.merge(
            base_df[["target_name", "probe_name", "prompt", "passed"]],
            mit_df[["target_name", "probe_name", "prompt", "passed"]],
            on=["target_name", "probe_name", "prompt"],
            suffixes=("_base", "_mit"),
        )

        if len(merged) > 0 and len(merged) == len(base_df) == len(mit_df):
            both_pass = int(((merged["passed_base"]) & (merged["passed_mit"])).sum())
            pass_to_fail = int(((merged["passed_base"]) & (~merged["passed_mit"])).sum())
            fail_to_pass = int(((~merged["passed_base"]) & (merged["passed_mit"])).sum())
            both_fail = int(((~merged["passed_base"]) & (~merged["passed_mit"])).sum())

            if pass_to_fail + fail_to_pass == 0:
                return 1.0

            table = np.array([[both_pass, pass_to_fail], [fail_to_pass, both_fail]])
            exact = (pass_to_fail + fail_to_pass) < 25
            res = mcnemar(table, exact=exact, correction=not exact)
            return float(res.pvalue)

        base_failed = int((~base_df["passed"]).sum())
        base_passed = int(base_df["passed"].sum())
        mit_failed = int((~mit_df["passed"]).sum())
        mit_passed = int(mit_df["passed"].sum())

        contingency = np.array([[base_failed, base_passed], [mit_failed, mit_passed]])
        if np.any(contingency.sum(axis=0) == 0) or np.any(contingency.sum(axis=1) == 0):
            return 1.0

        _, p_value, _, _ = chi2_contingency(contingency)
        return float(p_value)

    def compare(self) -> dict[str, Any]:
        """Compare baseline vs mitigated results."""
        base_df = pd.DataFrame([r.to_dict() for r in self.baseline])
        mitigated_df = pd.DataFrame([r.to_dict() for r in self.mitigated])

        if base_df.empty or mitigated_df.empty:
            return {
                "overall": {
                    "baseline_failure_rate": 0.0,
                    "mitigated_failure_rate": 0.0,
                    "absolute_risk_reduction": 0.0,
                    "relative_risk_reduction": 0.0,
                    "p_value": 1.0,
                    "effect_size": 0.0,
                    "significant": False,
                },
                "by_category": {},
                "regressions": [],
            }

        base_failed = int((~base_df["passed"]).sum())
        base_total = len(base_df)
        mit_failed = int((~mitigated_df["passed"]).sum())
        mit_total = len(mitigated_df)

        base_rate = base_failed / base_total
        mit_rate = mit_failed / mit_total
        p_value = self._compute_significance(base_df, mitigated_df)
        effect_size = float(proportion_effectsize(base_rate, mit_rate))

        categories = set(base_df["category"]) | set(mitigated_df["category"])
        category_comparison: dict[str, dict[str, float]] = {}

        for cat in sorted(categories):
            base_cat = base_df[base_df["category"] == cat]
            mit_cat = mitigated_df[mitigated_df["category"] == cat]

            if len(base_cat) > 0 and len(mit_cat) > 0:
                cat_base_rate = float((~base_cat["passed"]).mean())
                cat_mit_rate = float((~mit_cat["passed"]).mean())
                category_comparison[str(cat)] = {
                    "baseline_failure_rate": cat_base_rate,
                    "mitigated_failure_rate": cat_mit_rate,
                    "absolute_reduction": cat_base_rate - cat_mit_rate,
                    "relative_reduction": (
                        (cat_base_rate - cat_mit_rate) / cat_base_rate if cat_base_rate > 0 else 0.0
                    ),
                }

        abs_reduction = base_rate - mit_rate
        rel_reduction = abs_reduction / base_rate if base_rate > 0 else 0.0

        return {
            "overall": {
                "baseline_failure_rate": float(base_rate),
                "mitigated_failure_rate": float(mit_rate),
                "absolute_risk_reduction": float(abs_reduction),
                "relative_risk_reduction": float(rel_reduction),
                "p_value": float(p_value),
                "effect_size": effect_size,
                "significant": bool(p_value < 0.01),
            },
            "by_category": category_comparison,
            "regressions": self._find_regressions(base_df, mitigated_df),
        }

    def _find_regressions(
        self,
        base_df: pd.DataFrame,
        mit_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Find attacks that were blocked in baseline but succeed in mitigated."""
        regressions: list[dict[str, Any]] = []

        for _, base_row in base_df[base_df["passed"]].iterrows():
            mit_matches = mit_df[
                (mit_df["probe_name"] == base_row["probe_name"])
                & (mit_df["target_name"] == base_row["target_name"])
                & (mit_df["prompt"] == base_row["prompt"])
                & (~mit_df["passed"])
            ]

            if len(mit_matches) > 0:
                regressions.append(
                    {
                        "probe": base_row["probe_name"],
                        "target": base_row["target_name"],
                        "category": base_row["category"],
                        "severity": base_row["severity"],
                        "baseline_prompt": base_row["prompt"],
                        "mitigated_response": mit_matches.iloc[0]["response"],
                    }
                )

        return regressions


from test_harness.probes import CustomRAGProbe, create_baseline_probes  # noqa: E402

__all__ = [
    "AttackCategory",
    "AttackResult",
    "AttackSeverity",
    "ComparativeHarness",
    "CustomRAGProbe",
    "GarakProbeAdapter",
    "Probe",
    "PromptFooSuite",
    "PromptFooTestCase",
    "TestHarness",
    "create_baseline_probes",
]
