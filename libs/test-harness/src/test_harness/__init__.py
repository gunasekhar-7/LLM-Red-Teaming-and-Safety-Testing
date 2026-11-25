"""Test harness integrating Garak and PromptFoo for LLM red-teaming."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from rag_target import RAGTarget, RAGResponse
from llm_provider import LLMProvider, LLMRequest, ModelConfig, ProviderType


class AttackCategory(str, Enum):
    """Categories of attacks."""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    DATA_LEAKAGE = "data_leakage"
    RAG_SPECIFIC = "rag_specific"
    ENCODING = "encoding"
    ROLEPLAY = "roleplay"
    MULTITURN = "multiturn"


class AttackSeverity(str, Enum):
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
    timestamp: datetime = field(default_factory=datetime.utcnow)
    target_name: str = ""
    provider_name: str = ""
    probe_name: str = ""
    category: AttackCategory = AttackCategory.PROMPT_INJECTION
    severity: AttackSeverity = AttackSeverity.MEDIUM
    prompt: str = ""
    response: str = ""
    passed: bool = False  # True = attack blocked/failed, False = attack succeeded
    evidence: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "target_name": self.target_name,
            "provider_name": self.provider_name,
            "probe_name": self.probe_name,
            "category": self.category.value if hasattr(self.category, 'value') else self.category,
            "severity": self.severity.value if hasattr(self.severity, 'value') else self.severity,
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
        pass
    
    @abstractmethod
    async def evaluate(self, prompt: str, response: str, context: dict[str, Any] | None = None) -> bool:
        """Evaluate if attack succeeded. Returns True if attack FAILED (defended)."""
        pass
    
    async def run(
        self,
        target: RAGTarget,
        provider: LLMProvider | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackResult]:
        """Run this probe against a target."""
        prompts = await self.generate_prompts(context)
        results = []
        
        for prompt in prompts:
            start = time.perf_counter()
            try:
                rag_response: RAGResponse = await target.query(prompt)
                latency_ms = (time.perf_counter() - start) * 1000
                
                passed = await self.evaluate(prompt, rag_response.answer, context)
                
                results.append(AttackResult(
                    target_name=target.name,
                    provider_name=provider.config.model_name if provider else "unknown",
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
                ))
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                results.append(AttackResult(
                    target_name=target.name,
                    provider_name=provider.config.model_name if provider else "unknown",
                    probe_name=self.name,
                    category=self.category,
                    severity=self.severity,
                    prompt=prompt,
                    response="",
                    passed=True,  # Error = defended
                    evidence={},
                    latency_ms=latency_ms,
                    error=str(e),
                ))
        
        return results


class GarakProbeAdapter(Probe):
    """Adapter to run Garak probes through our harness."""
    
    def __init__(
        self,
        garak_probe_class: type,
        name: str | None = None,
        category: AttackCategory = AttackCategory.PROMPT_INJECTION,
        severity: AttackSeverity = AttackSeverity.MEDIUM,
        **garak_params,
    ):
        probe_name = name or garak_probe_class.__name__
        super().__init__(probe_name, category, severity)
        self.garak_probe_class = garak_probe_class
        self.garak_params = garak_params
        self._garak_probe = None
    
    async def _get_garak_probe(self):
        if self._garak_probe is None:
            self._garak_probe = self.garak_probe_class(**self.garak_params)
        return self._garak_probe
    
    async def generate_prompts(self, context: dict[str, Any] | None = None) -> list[str]:
        probe = await self._get_garak_probe()
        # Garak probes have a `prompts` property or `generate` method
        if hasattr(probe, "prompts"):
            return list(probe.prompts)
        elif hasattr(probe, "generate"):
            return await probe.generate()
        return []
    
    async def evaluate(self, prompt: str, response: str, context: dict[str, Any] | None = None) -> bool:
        probe = await self._get_garak_probe()
        # Garak uses detectors to evaluate
        if hasattr(probe, "detector") and probe.detector:
            return not probe.detector.detect(response)  # True if NOT detected (defended)
        return True  # Default to defended if no detector


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
    
    def __init__(
        self,
        targets: list[RAGTarget],
        providers: dict[str, LLMProvider] | None = None,
        probes: list[Probe] | None = None,
        output_dir: Path | str = "results",
    ):
        self.targets = targets
        self.providers = providers or {}
        self.probes = probes or []
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
        """Run all probes against all targets."""
        tasks = []
        
        for target in self.targets:
            for probe in self.probes:
                # Run with each provider if available
                if self.providers:
                    for provider_name, provider in self.providers.items():
                        tasks.append(self._run_probe_target(probe, target, provider))
                else:
                    tasks.append(self._run_probe_target(probe, target, None))
        
        if progress:
            results = await tqdm_asyncio.gather(*tasks, desc="Running probes")
        else:
            results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
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
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = filename or f"results_{timestamp}"
        
        jsonl_path = self.output_dir / f"{base}.jsonl"
        csv_path = self.output_dir / f"{base}.csv"
        
        # JSONL
        with open(jsonl_path, "w") as f:
            for result in self.results:
                f.write(json.dumps(result.to_dict()) + "\n")
        
        # CSV
        df = self.get_results_df()
        df.to_csv(csv_path, index=False)
        
        return jsonl_path
    
    def generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        df = self.get_results_df()
        
        summary = {
            "total_attacks": len(df),
            "total_passed": int(df["passed"].sum()),
            "total_failed": int((~df["passed"]).sum()),
            "overall_failure_rate": float((~df["passed"]).mean()),
            "by_category": df.groupby("category")["passed"].agg(["mean", "count"]).to_dict(),
            "by_target": df.groupby("target_name")["passed"].agg(["mean", "count"]).to_dict(),
            "by_provider": df.groupby("provider_name")["passed"].agg(["mean", "count"]).to_dict(),
            "by_probe": df.groupby("probe_name")["passed"].agg(["mean", "count"]).to_dict(),
            "by_severity": df.groupby("severity")["passed"].agg(["mean", "count"]).to_dict(),
            "avg_latency_ms": float(df["latency_ms"].mean()),
            "errors": int(df["error"].notna().sum()),
        }
        
        return summary
    
    def generate_report(self, template_path: Path | None = None) -> Path:
        """Generate HTML report from results."""
        from jinja2 import Environment, FileSystemLoader
        
        if template_path is None:
            template_path = Path(__file__).parent / "templates" / "report.html.j2"
        
        if not template_path.exists():
            # Create default template
            template_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_default_template(template_path)
        
        env = Environment(loader=FileSystemLoader(template_path.parent))
        template = env.get_template(template_path.name)
        
        summary = self.generate_summary()
        df = self.get_results_df()
        
        # Top failures
        failures = df[~df["passed"]].sort_values("severity", ascending=False).head(20)
        
        html = template.render(
            summary=summary,
            failures=failures.to_dict("records"),
            timestamp=datetime.utcnow().isoformat(),
            total_results=len(self.results),
        )
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"report_{timestamp}.html"
        
        with open(report_path, "w") as f:
            f.write(html)
        
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
        .chart { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>LLM Red-Teaming Security Assessment Report</h1>
    <p>Generated: {{ timestamp }}</p>
    <p>Total Tests: {{ total_results }}</p>
    
    <h2>Summary</h2>
    <div class="metric">
        <div class="metric-value">{{ "%.1f"|format(summary.overall_failure_rate * 100) }}%</div>
        <div class="metric-label">Overall Failure Rate</div>
    </div>
    <div class="metric">
        <div class="metric-value">{{ summary.total_failed }}</div>
        <div class="metric-label">Attacks Succeeded</div>
    </div>
    <div class="metric">
        <div class="metric-value">{{ summary.total_passed }}</div>
        <div class="metric-label">Attacks Blocked</div>
    </div>
    
    <h2>Failure Rate by Category</h2>
    <table>
        <tr><th>Category</th><th>Failure Rate</th><th>Count</th></tr>
        {% for cat, data in summary.by_category.mean.items() %}
        <tr>
            <td>{{ cat }}</td>
            <td class="{{ 'critical' if data > 0.5 else 'high' if data > 0.2 else 'medium' if data > 0.1 else 'low' }}">
                {{ "%.1f"|format(data * 100) }}%
            </td>
            <td>{{ summary.by_category.count[cat] }}</td>
        </tr>
        {% endfor %}
    </table>
    
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
            <td>{{ f.prompt[:100] }}...</td>
            <td>{{ f.response[:100] }}...</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>"""
        path.write_text(template)


class ComparativeHarness:
    """Harness for comparing baseline vs mitigated results."""
    
    def __init__(
        self,
        baseline_results: list[AttackResult],
        mitigated_results: list[AttackResult],
    ):
        self.baseline = baseline_results
        self.mitigated = mitigated_results
    
    def compare(self) -> dict[str, Any]:
        """Compare baseline vs mitigated results."""
        import scipy.stats as stats
        import numpy as np
        
        # Convert to DataFrames
        base_df = pd.DataFrame([r.to_dict() for r in self.baseline])
        mitigated_df = pd.DataFrame([r.to_dict() for r in self.mitigated])
        
        # Overall comparison
        base_failed = (~base_df["passed"]).sum()
        base_total = len(base_df)
        mit_failed = (~mitigated_df["passed"]).sum()
        mit_total = len(mitigated_df)
        
        # McNemar's test for paired proportions (if same prompts)
        # For now, use chi-squared test of independence
        from scipy.stats import chi2_contingency
        
        contingency = np.array([
            [base_failed, base_total - base_failed],
            [mit_failed, mit_total - mit_failed],
        ])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Effect size (Cohen's h)
        from statsmodels.stats.proportion import proportion_effectsize
        effect_size = proportion_effectsize(
            base_failed / base_total,
            mit_failed / mit_total,
        )
        
        # Category-level comparison
        categories = set(base_df["category"]) | set(mitigated_df["category"])
        category_comparison = {}
        
        for cat in categories:
            base_cat = base_df[base_df["category"] == cat]
            mit_cat = mitigated_df[mitigated_df["category"] == cat]
            
            if len(base_cat) > 0 and len(mit_cat) > 0:
                base_rate = (~base_cat["passed"]).mean()
                mit_rate = (~mit_cat["passed"]).mean()
                
                category_comparison[cat] = {
                    "baseline_failure_rate": float(base_rate),
                    "mitigated_failure_rate": float(mit_rate),
                    "absolute_reduction": float(base_rate - mit_rate),
                    "relative_reduction": float((base_rate - mit_rate) / base_rate) if base_rate > 0 else 0,
                }
        
        return {
            "overall": {
                "baseline_failure_rate": float(base_failed / base_total) if base_total > 0 else 0,
                "mitigated_failure_rate": float(mit_failed / mit_total) if mit_total > 0 else 0,
                "absolute_risk_reduction": float(base_failed / base_total - mit_failed / mit_total) if base_total > 0 and mit_total > 0 else 0,
                "relative_risk_reduction": float((base_failed / base_total - mit_failed / mit_total) / (base_failed / base_total)) if base_failed > 0 else 0,
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "significant": p_value < 0.01,
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
        regressions = []
        
        # Match by probe and target
        for _, base_row in base_df[base_df["passed"]].iterrows():
            mit_matches = mit_df[
                (mit_df["probe_name"] == base_row["probe_name"]) &
                (mit_df["target_name"] == base_row["target_name"]) &
                (~mit_df["passed"])
            ]
            
            if len(mit_matches) > 0:
                regressions.append({
                    "probe": base_row["probe_name"],
                    "target": base_row["target_name"],
                    "category": base_row["category"],
                    "severity": base_row["severity"],
                    "baseline_prompt": base_row["prompt"],
                    "mitigated_response": mit_matches.iloc[0]["response"],
                })
        
        return regressions