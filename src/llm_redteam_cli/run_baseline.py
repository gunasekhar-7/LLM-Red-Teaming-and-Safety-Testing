"""Baseline vulnerability assessment runner."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_target import RAGTarget
from test_harness import AttackResult, CustomRAGProbe, Probe, TestHarness, create_baseline_probes

__all__ = ["CustomRAGProbe", "create_baseline_probes", "main", "run_baseline"]


async def run_baseline(
    targets: Sequence[RAGTarget],
    providers: dict[str, Any] | None = None,
    output_dir: Path = Path("results/baseline"),
    max_concurrent: int = 3,
    progress: bool = True,
) -> tuple[list[AttackResult], Path]:
    """Run baseline vulnerability assessment."""
    probes: list[Probe] = create_baseline_probes()

    harness = TestHarness(
        targets=targets, providers=providers or {}, probes=probes, output_dir=output_dir
    )

    results = await harness.run_all(max_concurrent=max_concurrent, progress=progress)
    jsonl_path = harness.save_results("baseline")
    harness.generate_report()

    return results, jsonl_path


async def main() -> None:
    """Main entry point for baseline testing."""
    from existing_rag_adapter import MockExistingRAGAdapter
    from llm_provider import LLMSettings
    from llm_provider.factory import ProviderFactory

    targets: list[RAGTarget] = [
        MockExistingRAGAdapter("reference-rag"),
        MockExistingRAGAdapter("existing-rag"),
    ]

    settings = LLMSettings()
    factory = ProviderFactory(settings)
    providers = factory.create_default_providers()

    try:
        await run_baseline(targets, providers)
    finally:
        for target in targets:
            await target.close()
        for provider in providers.values():
            await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
