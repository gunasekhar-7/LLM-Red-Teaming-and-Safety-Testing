"""Comparative testing script for post-mitigation validation."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from test_harness import ComparativeHarness, TestHarness, AttackResult, AttackCategory, AttackSeverity
from .run_baseline import create_baseline_probes


def load_results(path: Path) -> list[AttackResult]:
    """Load results from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # Convert timestamp string back to datetime
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            # Convert enum strings back to enums
            data["category"] = AttackCategory(data["category"])
            data["severity"] = AttackSeverity(data["severity"])
            results.append(AttackResult(**data))
    return results


async def run_comparative(
    baseline_path: Path,
    targets: list[Any],
    providers: dict[str, Any] | None = None,
    output_dir: Path = Path("results/mitigated"),
) -> dict[str, Any]:
    """Run comparative test against baseline."""
    print(f"Loading baseline results from {baseline_path}...")
    baseline_results = load_results(baseline_path)
    print(f"Loaded {len(baseline_results)} baseline results")
    
    # Create same probes for fair comparison
    probes = create_baseline_probes()
    
    harness = TestHarness(
        targets=targets,
        providers=providers or {},
        probes=probes,
        output_dir=output_dir,
    )
    
    print("Running post-mitigation assessment...")
    mitigated_results = await harness.run_all(max_concurrent=3, progress=True)
    
    print(f"Completed {len(mitigated_results)} attack attempts")
    
    # Save mitigated results
    jsonl_path = harness.save_results("mitigated")
    report_path = harness.generate_report()
    
    # Compare
    print("Comparing results...")
    comparator = ComparativeHarness(baseline_results, mitigated_results)
    comparison = comparator.compare()
    
    # Save comparison
    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # Print summary
    print("\n=== COMPARISON SUMMARY ===")
    overall = comparison["overall"]
    print(f"Baseline failure rate: {overall['baseline_failure_rate']:.1%}")
    print(f"Mitigated failure rate: {overall['mitigated_failure_rate']:.1%}")
    print(f"Absolute reduction: {overall['absolute_risk_reduction']:.1%}")
    print(f"Relative reduction: {overall['relative_risk_reduction']:.1%}")
    print(f"P-value: {overall['p_value']:.6f}")
    print(f"Effect size (Cohen's h): {overall['effect_size']:.3f}")
    print(f"Significant (p<0.01): {overall['significant']}")
    
    print("\nBy Category:")
    for cat, data in comparison["by_category"].items():
        print(f"  {cat}: {data['baseline_failure_rate']:.1%} -> {data['mitigated_failure_rate']:.1%} "
              f"(Delta={data['absolute_reduction']:.1%}, rel={data['relative_reduction']:.1%})")
    
    if comparison["regressions"]:
        print(f"\n[WARNING] REGRESSIONS FOUND: {len(comparison['regressions'])}")
        for reg in comparison["regressions"][:5]:
            print(f"  - {reg['probe']} on {reg['target']}: was blocked, now succeeds")
    else:
        print("\n[OK] No regressions detected")
    
    print(f"\nComparison saved to: {comparison_path}")
    print(f"Mitigated report: {report_path}")
    
    return comparison


async def main():
    """Main entry point for comparative testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comparative red-teaming test")
    parser.add_argument("baseline", type=Path, help="Path to baseline results JSONL")
    parser.add_argument("--output", type=Path, default=Path("results/mitigated"), help="Output directory")
    args = parser.parse_args()
    
    if not args.baseline.exists():
        print(f"Baseline file not found: {args.baseline}")
        return
    
    # Initialize targets (same as baseline)
    from existing_rag_adapter import MockExistingRAGAdapter
    
    reference_target = MockExistingRAGAdapter("reference-rag")
    existing_target = MockExistingRAGAdapter("existing-rag")
    
    targets = [reference_target, existing_target]
    
    try:
        await run_comparative(args.baseline, targets, output_dir=args.output)
    finally:
        await reference_target.close()
        await existing_target.close()


if __name__ == "__main__":
    asyncio.run(main())