"""Thin report shim delegating to llm_redteam_cli.generate_report."""

from llm_redteam_cli.generate_report import AuditReportGenerator, generate_report_from_files, main

__all__ = ["AuditReportGenerator", "generate_report_from_files", "main"]

if __name__ == "__main__":
    main()
