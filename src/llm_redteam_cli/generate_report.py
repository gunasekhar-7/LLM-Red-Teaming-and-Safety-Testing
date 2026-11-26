"""Professional audit report generator."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from test_harness import AttackResult, TestHarness, create_baseline_probes

from llm_redteam_cli.run_comparative import load_results


class AuditReportGenerator:
    """Generate professional security audit reports in HTML or Markdown."""

    def __init__(self, template_dir: Path | None = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.html_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(["html", "xml", "j2"]),
        )
        self.text_env = Environment(loader=FileSystemLoader(self.template_dir), autoescape=False)

    def generate(
        self,
        baseline_results: list[AttackResult],
        comparison: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        output_path: Path | None = None,
        format: str = "html",
    ) -> Path:
        """Generate audit report in the requested format ('html' or 'md')."""
        if format not in {"html", "md"}:
            raise ValueError(f"Unsupported report format: {format}")

        harness = TestHarness(targets=[], probes=[])
        harness.results = baseline_results
        summary = harness.generate_summary()

        policy_path = Path("config/mitigation_policies.yaml")
        full_config = (
            policy_path.read_text(encoding="utf-8")
            if policy_path.exists()
            else "# See config/mitigation_policies.yaml"
        )

        template_data: dict[str, Any] = {
            "metadata": metadata
            or {
                "title": "LLM Red-Teaming Security Audit Report",
                "version": "1.0.0",
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "classification": "CONFIDENTIAL",
                "targets": sorted({r.target_name for r in baseline_results}),
            },
            "executive_summary": (
                f"Automated red-teaming assessment evaluated {summary.get('total_attacks', 0)} "
                f"attack prompts with a baseline failure rate of "
                f"{summary.get('overall_failure_rate', 0.0):.1%}."
            ),
            "summary": summary,
            "validation": comparison,
            "scope": {
                "description": (
                    "Security assessment of RAG-based LLM applications against prompt injection, "
                    "data leakage, jailbreaking, and RAG-specific vulnerabilities."
                ),
                "targets": sorted({r.target_name for r in baseline_results}),
                "providers": sorted({r.provider_name for r in baseline_results}),
                "categories": sorted(
                    {
                        r.category.value if hasattr(r.category, "value") else str(r.category)
                        for r in baseline_results
                    }
                ),
                "duration": "Automated",
            },
            "methodology": (
                "Automated red-teaming using Garak probes, PromptFoo test suites, and custom "
                "RAG-specific attack vectors. Statistical validation via paired McNemar's test "
                "(with chi-squared fallback) and Cohen's h effect size."
            ),
            "architecture": {
                "description": (
                    "Multi-tenant RAG application with LangChain orchestration, pgvector for "
                    "embeddings, and configurable LLM providers."
                ),
                "components": [
                    {
                        "name": "API Gateway",
                        "tech": "FastAPI",
                        "version": "0.109+",
                        "config": "JWT/API Key auth, rate limiting",
                    },
                    {
                        "name": "RAG Orchestrator",
                        "tech": "LangChain",
                        "version": "0.1+",
                        "config": "Custom chain with retriever + generator",
                    },
                    {
                        "name": "Vector Store",
                        "tech": "PostgreSQL + pgvector",
                        "version": "16/0.5",
                        "config": "IVFFlat index, cosine similarity",
                    },
                    {
                        "name": "LLM Providers",
                        "tech": "OpenAI/Anthropic/Ollama",
                        "version": "Multi",
                        "config": "gpt-4o-mini, claude-3.5-haiku, llama3.1:8b",
                    },
                    {
                        "name": "Embeddings",
                        "tech": "OpenAI/Ollama",
                        "version": "Multi",
                        "config": "text-embedding-3-small, nomic-embed-text",
                    },
                ],
            },
            "top_findings": sorted(
                [r for r in baseline_results if not r.passed],
                key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(
                    x.severity.value if hasattr(x.severity, "value") else str(x.severity), 5
                ),
            ),
            "mitigations": {
                "overview": (
                    "Three-layer defense-in-depth: Input filtering, Retrieval sanitation, "
                    "and Output sanitization."
                ),
                "layers": [
                    {
                        "name": "Input Guard",
                        "components": [
                            "Injection Classifier (DeBERTa)",
                            "Pattern Blocking",
                            "Encoding Detection",
                            "Length Normalization",
                        ],
                        "coverage": "Prompt Injection, Encoding",
                        "latency": "< 50ms p95",
                    },
                    {
                        "name": "Retrieval Guard",
                        "components": [
                            "PII Redaction (Presidio)",
                            "Relevance Filtering",
                            "Tenant Isolation",
                        ],
                        "coverage": "Data Leakage, Cross-tenant",
                        "latency": "< 20ms p95",
                    },
                    {
                        "name": "Output Guard",
                        "components": [
                            "Citation Verification",
                            "PII Redaction",
                            "Refusal Detection",
                            "Length Enforcement",
                        ],
                        "coverage": "Citation Forgery, Data Leakage",
                        "latency": "< 30ms p95",
                    },
                ],
                "config_yaml": full_config,
            },
            "residual_risks": [
                {
                    "id": "RISK-001",
                    "category": "Prompt Injection",
                    "likelihood": "Low",
                    "impact": "High",
                    "score": "Medium",
                    "score_class": "medium",
                    "status": "Monitored",
                },
                {
                    "id": "RISK-002",
                    "category": "Novel Jailbreaks",
                    "likelihood": "Medium",
                    "impact": "High",
                    "score": "Medium",
                    "score_class": "medium",
                    "status": "Continuous Testing",
                },
                {
                    "id": "RISK-003",
                    "category": "Multi-turn Attacks",
                    "likelihood": "Medium",
                    "impact": "Medium",
                    "score": "Medium",
                    "score_class": "medium",
                    "status": "Partial Coverage",
                },
                {
                    "id": "RISK-004",
                    "category": "Encoding Variants",
                    "likelihood": "Low",
                    "impact": "Medium",
                    "score": "Low",
                    "score_class": "low",
                    "status": "Monitored",
                },
            ],
            "recommendations": [
                {
                    "title": "Continuous Red-Teaming",
                    "description": "Integrate automated testing into CI/CD pipeline for regression detection.",
                },
                {
                    "title": "Expand Probe Coverage",
                    "description": "Add multi-turn conversation probes, token smuggling, and model-specific bypasses.",
                },
                {
                    "title": "Adaptive Defense",
                    "description": "Tune classifier thresholds and patterns against observed production anomalies.",
                },
                {
                    "title": "Enhanced Monitoring",
                    "description": "Alert on anomalous block rates and cross-tenant access attempts.",
                },
                {
                    "title": "Regular Re-assessment",
                    "description": "Schedule periodic re-testing against updated threat vectors.",
                },
            ],
            "probe_catalog": [
                {
                    "name": p.name,
                    "category": p.category.value,
                    "severity": p.severity.value,
                    "description": p.description,
                }
                for p in create_baseline_probes()
            ],
            "full_config": full_config,
            "enumerate": enumerate,
            "severity_for_category": lambda cat: {
                "prompt_injection": "High",
                "jailbreak": "High",
                "data_leakage": "Critical",
                "rag_specific": "High",
                "encoding": "Medium",
                "roleplay": "Medium",
                "multiturn": "Medium",
            }.get(cat, "Medium"),
        }

        template_name = f"audit_report.{format}.j2"
        env = self.html_env if format == "html" else self.text_env
        template = env.get_template(template_name)
        rendered = template.render(**template_data)

        if output_path is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"audit_report_{timestamp}.{format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        return output_path


def generate_report_from_files(
    results_path: Path,
    comparison_path: Path | None = None,
    format: str = "html",
    output_path: Path | None = None,
) -> Path:
    """Load result artifacts and generate an audit report."""
    results = load_results(results_path)
    comparison = None
    if comparison_path and comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    generator = AuditReportGenerator()
    return generator.generate(
        baseline_results=results, comparison=comparison, output_path=output_path, format=format
    )


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate audit report")
    parser.add_argument("results", type=Path, help="Baseline results JSONL")
    parser.add_argument("--comparison", type=Path, help="Comparison JSON")
    parser.add_argument("--format", choices=["html", "md"], default="html")
    parser.add_argument("--output", type=Path, help="Output file path")
    args = parser.parse_args()

    output = generate_report_from_files(
        results_path=args.results,
        comparison_path=args.comparison,
        format=args.format,
        output_path=args.output,
    )
    print(f"Report generated: {output}")


if __name__ == "__main__":
    main()
