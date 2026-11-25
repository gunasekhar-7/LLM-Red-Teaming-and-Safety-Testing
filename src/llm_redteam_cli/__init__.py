"""Main CLI for LLM Red-Teaming Framework."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from test_harness import TestHarness, AttackCategory, AttackSeverity
from .run_baseline import create_baseline_probes, run_baseline
from .run_comparative import run_comparative
from .generate_report import AuditReportGenerator, main as generate_report_main


app = typer.Typer(
    name="llm-redteam",
    help="LLM Red-Teaming Framework for RAG Applications",
    add_completion=False,
)
console = Console()


@app.command()
def baseline(
    targets: Annotated[list[str], typer.Option("--target", "-t", help="Target names to test")] = ["reference-rag", "existing-rag"],
    providers: Annotated[list[str], typer.Option("--provider", "-p", help="Providers to test")] = [],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("results/baseline"),
    concurrent: Annotated[int, typer.Option("--concurrent", "-c", help="Max concurrent tests")] = 3,
):
    """Run baseline vulnerability assessment."""
    console.print("[bold blue]Running Baseline Vulnerability Assessment[/bold blue]")
    
    # Initialize targets
    from existing_rag_adapter import MockExistingRAGAdapter
    
    target_instances = []
    for name in targets:
        if name == "reference-rag":
            target_instances.append(MockExistingRAGAdapter("reference-rag"))
        elif name == "existing-rag":
            target_instances.append(MockExistingRAGAdapter("existing-rag"))
        else:
            console.print(f"[yellow]Unknown target: {name}[/yellow]")
    
    # Initialize providers if specified
    provider_instances = {}
    if providers:
        from llm_provider import ProviderFactory, LLMSettings
        settings = LLMSettings()
        factory = ProviderFactory(settings)
        all_providers = factory.create_default_providers()
        for p in providers:
            if p in all_providers:
                provider_instances[p] = all_providers[p]
            else:
                console.print(f"[yellow]Provider not available: {p}[/yellow]")
    
    async def _run():
        try:
            results, path = await run_baseline(
                targets=target_instances,
                providers=provider_instances,
                output_dir=output,
            )
            
            # Print summary table
            table = Table(title="Baseline Results Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            from test_harness import TestHarness
            harness = TestHarness(targets=[], probes=[])
            harness.results = results
            summary = harness.generate_summary()
            
            table.add_row("Total Attacks", str(summary["total_attacks"]))
            table.add_row("Blocked", str(summary["total_passed"]))
            table.add_row("Succeeded", str(summary["total_failed"]))
            table.add_row("Failure Rate", f"{summary['overall_failure_rate']:.1%}")
            table.add_row("Avg Latency", f"{summary['avg_latency_ms']:.0f}ms")
            
            console.print(table)
            console.print(f"\n[green]Results saved to: {path}[/green]")
            
        finally:
            for target in target_instances:
                await target.close()
            for provider in provider_instances.values():
                await provider.close()
    
    asyncio.run(_run())


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Path to baseline results JSONL")],
    targets: Annotated[list[str], typer.Option("--target", "-t", help="Target names to test")] = ["reference-rag", "existing-rag"],
    providers: Annotated[list[str], typer.Option("--provider", "-p", help="Providers to test")] = [],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("results/mitigated"),
):
    """Run comparative test against baseline."""
    console.print("[bold blue]Running Comparative Assessment[/bold blue]")
    
    if not baseline.exists():
        console.print(f"[red]Baseline file not found: {baseline}[/red]")
        raise typer.Exit(1)
    
    from existing_rag_adapter import MockExistingRAGAdapter
    
    target_instances = []
    for name in targets:
        if name == "reference-rag":
            target_instances.append(MockExistingRAGAdapter("reference-rag"))
        elif name == "existing-rag":
            target_instances.append(MockExistingRAGAdapter("existing-rag"))
    
    provider_instances = {}
    if providers:
        from llm_provider import ProviderFactory, LLMSettings
        settings = LLMSettings()
        factory = ProviderFactory(settings)
        all_providers = factory.create_default_providers()
        for p in providers:
            if p in all_providers:
                provider_instances[p] = all_providers[p]
    
    async def _run():
        try:
            comparison = await run_comparative(
                baseline_path=baseline,
                targets=target_instances,
                providers=provider_instances,
                output_dir=output,
            )
            
            # Print comparison table
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
            table.add_row("Relative Reduction", "-", "-", f"{overall['relative_risk_reduction']:.1%}")
            table.add_row("P-value", "-", "-", f"{overall['p_value']:.6f}")
            table.add_row("Significant", "-", "-", "YES" if overall['significant'] else "NO")
            
            console.print(table)
            
            # Category breakdown
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
                console.print(f"\n[red][WARNING] {len(comparison['regressions'])} REGRESSIONS DETECTED[/red]")
            else:
                console.print("\n[green][OK] No regressions detected[/green]")
            
        finally:
            for target in target_instances:
                await target.close()
            for provider in provider_instances.values():
                await provider.close()
    
    asyncio.run(_run())


@app.command()
def report(
    results: Annotated[Path, typer.Argument(help="Baseline results JSONL")],
    comparison: Annotated[Path | None, typer.Option("--comparison", "-c", help="Comparison JSON")] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="Output format")] = "html",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file")] = None,
):
    """Generate professional audit report."""
    console.print("[bold blue]Generating Audit Report[/bold blue]")
    
    import sys
    sys.argv = ["generate_report", str(results)]
    if comparison:
        sys.argv.extend(["--comparison", str(comparison)])
    sys.argv.extend(["--format", format])
    if output:
        sys.argv.extend(["--output", str(output)])
    
    asyncio.run(generate_report_main())
    console.print("[green]Report generated successfully[/green]")


@app.command()
def dashboard(
    host: Annotated[str, typer.Option("--host", help="Host to bind")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="Port to bind")] = 8501,
):
    """Launch interactive Streamlit dashboard."""
    console.print(f"[bold blue]Starting dashboard on http://{host}:{port}[/bold blue]")
    import subprocess
    subprocess.run(["streamlit", "run", "dashboard/app.py", "--server.address", host, "--server.port", str(port)])


@app.command()
def list_probes():
    """List available attack probes."""
    probes = create_baseline_probes()
    
    table = Table(title="Available Probes")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Severity", style="red")
    table.add_column("Description")
    table.add_column("Prompts", style="dim")
    
    for probe in probes:
        table.add_row(
            probe.name,
            probe.category.value,
            probe.severity.value,
            probe.description,
            str(len(probe._prompts)) if hasattr(probe, "_prompts") else "N/A",
        )
    
    console.print(table)


@app.command()
def init_db():
    """Initialize database for reference RAG."""
    console.print("[bold blue]Initializing database...[/bold blue]")
    from reference_rag.database import init_db
    asyncio.run(init_db())
    console.print("[green]Database initialized[/green]")


if __name__ == "__main__":
    app()