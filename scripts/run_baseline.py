"""Thin baseline shim delegating to llm_redteam_cli.run_baseline."""

import asyncio

from llm_redteam_cli.run_baseline import CustomRAGProbe, create_baseline_probes, main, run_baseline

__all__ = ["CustomRAGProbe", "create_baseline_probes", "main", "run_baseline"]

if __name__ == "__main__":
    asyncio.run(main())
