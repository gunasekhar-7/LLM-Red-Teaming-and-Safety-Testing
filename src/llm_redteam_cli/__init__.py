"""Main CLI for LLM Red-Teaming Framework."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from existing_rag_adapter import (
    ExistingRAGAdapter,
    ExistingRAGConfig,
    HardenedExistingRAGAdapter,
    MockExistingRAGAdapter,
)
from rag_target import BaseRAGTarget
from rich.console import Console
from rich.table import Table
from test_harness import TestHarness

from llm_redteam_cli.generate_report import generate_report_from_files
from llm_redteam_cli.run_baseline import create_baseline_probes, run_baseline
from llm_redteam_cli.run_comparative import run_comparative

app = typer.Typer(
    name="llm-redteam", help="LLM Red-Teaming Framework for RAG Applications", add_completion=False
)
console = Console()


def _build_targets(
    targets: list[str],
    mock: bool = False,
    hardened: bool = False,
    reference_rag_url: str | None = None,
    existing_rag_url: str | None = None,
) -> list[BaseRAGTarget]:
    """Build RAGTarget instances from CLI options / environment."""
    ref_url = reference_rag_url or os.getenv("REFERENCE_RAG_URL")
    ext_url = existing_rag_url or os.getenv("EXISTING_RAG_URL")

    instances: list[BaseRAGTarget] = []
    for name in targets:
        target: BaseRAGTarget
        if name == "reference-rag":
            if not mock and ref_url:
                target = ExistingRAGAdapter(
                    "reference-rag", ExistingRAGConfig(api_base_url=ref_url)
                )
            else:
                target = MockExistingRAGAdapter("reference-rag")
        elif name == "existing-rag":
            if not mock and ext_url:
                target = ExistingRAGAdapter("existing-rag", ExistingRAGConfig(api_base_url=ext_url))
            else:
                target = MockExistingRAGAdapter("existing-rag")
        else:
            console.print(f"[yellow]Unknown target: {name}[/yellow]")
            continue

        if hardened:
            target = HardenedExistingRAGAdapter(f"{name}-hardened", target)
        instances.append(target)

    return instances


def _build_providers(providers: list[str]) -> dict[str, object]:
    """Build requested LLM provider instances."""
    provider_instances: dict[str, object] = {}
    if not providers:
        return provider_instances

    from llm_provider import LLMSettings
    from llm_provider.factory import ProviderFactory

    settings = LLMSettings()
    factory = ProviderFactory(settings)
    all_providers = factory.create_default_providers()
    for name in providers:
        if name in all_providers:
            provider_instances[name] = all_providers[name]
        else:
            console.print(f"[yellow]Provider not available: {name}[/yellow]")
    return provider_instances


@app.command()
def baseline(
    targets: Annotated[
        list[str] | None, typer.Option("--target", "-t", help="Target names to test")
    ] = None,
    providers: Annotated[
        list[str] | None, typer.Option("--provider", "-p", help="Providers to test")
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "results/baseline"
    ),
    concurrent: Annotated[int, typer.Option("--concurrent", "-c", help="Max concurrent tests")] = 3,
    mock: Annotated[
        bool, typer.Option("--mock/--live", help="Use mock adapters instead of live URLs")
    ] = False,
    hardened: Annotated[
        bool, typer.Option("--hardened", help="Wrap targets with defense guard pipelines")
    ] = False,
    reference_rag_url: Annotated[
        str | None, typer.Option("--reference-rag-url", help="Live Reference RAG base URL")
    ] = None,
    existing_rag_url: Annotated[
        str | None, typer.Option("--existing-rag-url", help="Live Existing RAG base URL")
    ] = None,
) -> None:
    """Run baseline vulnerability assessment."""
    console.print("[bold blue]Running Baseline Vulnerability Assessment[/bold blue]")

    target_instances = _build_targets(
        targets or ["reference-rag", "existing-rag"],
        mock=mock,
        hardened=hardened,
        reference_rag_url=reference_rag_url,
        existing_rag_url=existing_rag_url,
    )
    provider_instances = _build_providers(providers or [])

    async def _run() -> None:
        try:
            results, path = await run_baseline(
                targets=target_instances,
                providers=provider_instances,
                output_dir=output,
                max_concurrent=concurrent,
            )

            table = Table(title="Baseline Results Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            harness = TestHarness(targets=[], probes=[])
            harness.results = results
            summary = harness.generate_summary()

            table.add_row("Total Attacks", str(summary.get("total_attacks", 0)))
            table.add_row("Blocked", str(summary.get("total_passed", 0)))
            table.add_row("Succeeded", str(summary.get("total_failed", 0)))
            table.add_row("Failure Rate", f"{summary.get('overall_failure_rate', 0.0):.1%}")
            table.add_row("Avg Latency", f"{summary.get('avg_latency_ms', 0.0):.0f}ms")

            console.print(table)
            console.print(f"\n[green]Results saved to: {path}[/green]")
        finally:
            for target in target_instances:
                await target.close()
            for provider in provider_instances.values():
                await provider.close()  # type: ignore[attr-defined]

    asyncio.run(_run())


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Path to baseline results JSONL")],
    targets: Annotated[
        list[str] | None, typer.Option("--target", "-t", help="Target names to test")
    ] = None,
    providers: Annotated[
        list[str] | None, typer.Option("--provider", "-p", help="Providers to test")
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "results/mitigated"
    ),
    concurrent: Annotated[int, typer.Option("--concurrent", "-c", help="Max concurrent tests")] = 3,
    mock: Annotated[
        bool, typer.Option("--mock/--live", help="Use mock adapters instead of live URLs")
    ] = False,
    hardened: Annotated[
        bool, typer.Option("--hardened", help="Wrap targets with defense guard pipelines")
    ] = True,
) -> None:
    """Run comparative test against baseline."""
    console.print("[bold blue]Running Comparative Assessment[/bold blue]")

    if not baseline.exists():
        console.print(f"[red]Baseline file not found: {baseline}[/red]")
        raise typer.Exit(1)

    target_instances = _build_targets(
        targets or ["reference-rag", "existing-rag"], mock=mock, hardened=hardened
    )
    provider_instances = _build_providers(providers or [])

    async def _run() -> None:
        try:
            comparison = await run_comparative(
                baseline_path=baseline,
                targets=target_instances,
                providers=provider_instances,
                output_dir=output,
                max_concurrent=concurrent,
            )

            table = Table(title="Comparison Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Baseline", style="yellow")
            table.add_column("Mitigated", style="green")
            table.add_column("Change", style="bold")

            overall = comparison["overall"]
            table.add_row(
                "Failure Rate",
                f"{overall['baseline_failure_rate']:.1%}",
                f"{overall['mitigated_failure_rate']:.1%}",
                f"{overall['absolute_risk_reduction']:+.1%}",
            )
            table.add_row(
                "Relative Reduction", "-", "-", f"{overall['relative_risk_reduction']:.1%}"
            )
            table.add_row("P-value", "-", "-", f"{overall['p_value']:.6f}")
            table.add_row("Significant", "-", "-", "YES" if overall["significant"] else "NO")
            console.print(table)

            cat_table = Table(title="By Category")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Baseline", style="yellow")
            cat_table.add_column("Mitigated", style="green")
            cat_table.add_column("Abs. Delta", style="bold")
            cat_table.add_column("Rel. Delta", style="bold")

            for cat, data in comparison["by_category"].items():
                cat_table.add_row(
                    cat.replace("_", " ").title(),
                    f"{data['baseline_failure_rate']:.1%}",
                    f"{data['mitigated_failure_rate']:.1%}",
                    f"{data['absolute_reduction']:+.1%}",
                    f"{data['relative_reduction']:.1%}",
                )
            console.print(cat_table)

            if comparison["regressions"]:
                console.print(
                    f"\n[red][WARNING] {len(comparison['regressions'])} REGRESSIONS DETECTED[/red]"
                )
            else:
                console.print("\n[green][OK] No regressions detected[/green]")
        finally:
            for target in target_instances:
                await target.close()
            for provider in provider_instances.values():
                await provider.close()  # type: ignore[attr-defined]

    asyncio.run(_run())


@app.command()
def report(
    results: Annotated[Path, typer.Argument(help="Baseline results JSONL")],
    comparison: Annotated[
        Path | None, typer.Option("--comparison", "-c", help="Comparison JSON")
    ] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="Output format")] = "html",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file")] = None,
) -> None:
    """Generate professional audit report."""
    console.print("[bold blue]Generating Audit Report[/bold blue]")
    if not results.exists():
        console.print(f"[red]Results file not found: {results}[/red]")
        raise typer.Exit(1)

    report_path = generate_report_from_files(
        results_path=results, comparison_path=comparison, format=format, output_path=output
    )
    console.print(f"[green]Report generated: {report_path}[/green]")


@app.command()
def dashboard(
    host: Annotated[str, typer.Option("--host", help="Host to bind")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="Port to bind")] = 8501,
) -> None:
    """Launch interactive Streamlit dashboard."""
    console.print(f"[bold blue]Starting dashboard on http://{host}:{port}[/bold blue]")
    subprocess.run(
        [
            "streamlit",
            "run",
            "dashboard/app.py",
            "--server.address",
            host,
            "--server.port",
            str(port),
        ],
        check=False,
    )


@app.command()
def list_probes() -> None:
    """List available attack probes."""
    probes = create_baseline_probes()

    table = Table(title="Available Probes")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Severity", style="red")
    table.add_column("Description")
    table.add_column("Prompts", style="dim")

    for probe in probes:
        prompt_count = str(len(probe._prompts)) if hasattr(probe, "_prompts") else "N/A"
        table.add_row(
            probe.name, probe.category.value, probe.severity.value, probe.description, prompt_count
        )

    console.print(table)


@app.command()
def init_db() -> None:
    """Initialize database for reference RAG."""
    console.print("[bold blue]Initializing database...[/bold blue]")
    from reference_rag.database import init_db as _init_db

    asyncio.run(_init_db())
    console.print("[green]Database initialized[/green]")


if __name__ == "__main__":
    app()
