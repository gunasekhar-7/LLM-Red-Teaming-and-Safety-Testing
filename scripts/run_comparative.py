"""Thin comparative shim delegating to llm_redteam_cli.run_comparative."""

import asyncio

from llm_redteam_cli.run_comparative import load_results, main, run_comparative

__all__ = ["load_results", "main", "run_comparative"]

if __name__ == "__main__":
    asyncio.run(main())
