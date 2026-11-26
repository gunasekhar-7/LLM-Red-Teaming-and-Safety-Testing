"""Comparative testing runner for post-mitigation validation."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from test_harness import (
    AttackCategory,
    AttackResult,
    AttackSeverity,
    ComparativeHarness,
    TestHarness,
    create_baseline_probes,
)


def load_results(path: Path) -> list[AttackResult]:
    """Load results from JSONL file."""
    results: list[AttackResult] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            data["category"] = AttackCategory(data["category"])
            data["severity"] = AttackSeverity(data["severity"])
            results.append(AttackResult(**data))
    return results


async def run_comparative(
    baseline_path: Path,
    targets: list[Any],
    providers: dict[str, Any] | None = None,
    output_dir: Path = Path("results/mitigated"),
    max_concurrent: int = 3,
    progress: bool = True,
) -> dict[str, Any]:
    """Run comparative test against baseline."""
    baseline_results = load_results(baseline_path)
    probes = create_baseline_probes()

    harness = TestHarness(
        targets=targets, providers=providers or {}, probes=probes, output_dir=output_dir
    )

    mitigated_results = await harness.run_all(max_concurrent=max_concurrent, progress=progress)
    harness.save_results("mitigated")
    harness.generate_report()

    comparator = ComparativeHarness(baseline_results, mitigated_results)
    comparison = comparator.compare()

    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    return comparison


async def main() -> None:
    """Main entry point for comparative testing."""
    import argparse

    from existing_rag_adapter import MockExistingRAGAdapter

    parser = argparse.ArgumentParser(description="Run comparative red-teaming test")
    parser.add_argument("baseline", type=Path, help="Path to baseline results JSONL")
    parser.add_argument("--output", type=Path, default=Path("results/mitigated"))
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"Baseline file not found: {args.baseline}")
        return

    targets = [MockExistingRAGAdapter("reference-rag"), MockExistingRAGAdapter("existing-rag")]
    try:
        await run_comparative(args.baseline, targets, output_dir=args.output)
    finally:
        for target in targets:
            await target.close()


if __name__ == "__main__":
    asyncio.run(main())
