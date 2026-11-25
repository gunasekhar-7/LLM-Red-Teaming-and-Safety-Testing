# Changelog

All notable changes to this project will be documented in this format.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with monorepo layout
- Core RAGTarget protocol and base classes
- Multi-provider LLM abstraction (OpenAI, Anthropic, Ollama, LiteLLM)
- Defense guard layers (Input, Retrieval, Output)
- Test harness integrating Garak and PromptFoo
- Reference RAG application (FastAPI + pgvector + LangChain)
- Existing RAG adapter (HTTP and direct LangChain modes)
- Baseline vulnerability assessment suite (50+ probes)
- Comparative testing with statistical validation
- Professional audit report generator (HTML/Markdown)
- Interactive Streamlit dashboard
- Docker Compose infrastructure (PostgreSQL, Ollama, Prometheus, Grafana)
- Comprehensive documentation (README, ARCHITECTURE, SECURITY, CONTRIBUTING)

### Security
- Secret scanning in pre-commit hooks
- Responsible disclosure policy
- Attack data handling guidelines

## [0.1.0] - 2026-11-03

### Added
- Initial release of LLM Red-Teaming Framework
- Support for prompt injection, jailbreaking, data leakage, RAG-specific attacks
- Multi-provider testing (OpenAI, Anthropic, Ollama)
- Three-layer defense-in-depth (Input, Retrieval, Output guards)
- Statistical validation (McNemar's test, Cohen's h)
- Regression detection
- Professional reporting

---

## Release Process

### Version Numbering
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create release branch: `git checkout -b release/vX.Y.Z`
4. Run full test suite: `uv run pytest`
5. Build Docker images: `docker compose -f infra/docker/docker-compose.yml build`
6. Create PR to main
7. After merge, tag release: `git tag vX.Y.Z`
8. GitHub Actions publishes to container registry
9. Update documentation site