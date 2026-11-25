# Contributing Guide

Thank you for contributing to the LLM Red-Teaming Framework!

## Getting Started

### Prerequisites

- Python 3.11+
- UV package manager (`pip install uv`)
- Docker & Docker Compose
- Git

### Development Setup

```bash
# Fork and clone
git clone https://github.com/your-org/llm-redteaming.git
cd llm-redteaming

# Create virtual environment with UV
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install all dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Start infrastructure
cd infra/docker && docker compose up -d
```

## Code Standards

### Style Guide

- **Formatter**: Ruff (line length: 100)
- **Type Checking**: MyPy (strict mode)
- **Linting**: Ruff with strict rules
- **Imports**: Absolute imports, sorted by Ruff

### Code Quality

```bash
# Run all checks
uv run ruff check .
uv run ruff format .
uv run mypy .

# Run tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

Examples:
```
feat(probes): add multi-turn conversation probe
fix(guards): correct PII redaction for credit cards
docs(readme): update quick start instructions
test(harness): add regression detection tests
```

## Adding New Probes

### 1. Create Probe Class

```python
# libs/test-harness/src/test_harness/probes/my_probe.py
from test_harness import Probe, AttackCategory, AttackSeverity

class MyCustomProbe(Probe):
    def __init__(self):
        super().__init__(
            name="my_custom_probe",
            category=AttackCategory.RAG_SPECIFIC,
            severity=AttackSeverity.HIGH,
            description="Description of what this probe tests",
        )
    
    async def generate_prompts(self, context=None):
        return [
            "Attack prompt 1",
            "Attack prompt 2",
        ]
    
    async def evaluate(self, prompt, response, context=None):
        # Return True if DEFENDED, False if attack SUCCEEDED
        return "refusal indicator" in response.lower()
```

### 2. Register in Probe Suite

```python
# scripts/run_baseline.py
from test_harness.probes.my_probe import MyCustomProbe

def create_baseline_probes():
    probes = [...]
    probes.append(MyCustomProbe())
    return probes
```

### 3. Add Tests

```python
# tests/unit/test_my_probe.py
import pytest
from test_harness.probes.my_probe import MyCustomProbe

@pytest.mark.asyncio
async def test_my_probe_generates_prompts():
    probe = MyCustomProbe()
    prompts = await probe.generate_prompts()
    assert len(prompts) == 2

@pytest.mark.asyncio
async def test_my_probe_evaluates_correctly():
    probe = MyCustomProbe()
    # Test defended case
    assert await probe.evaluate("", "I cannot help with that") is True
    # Test successful attack
    assert await probe.evaluate("", "Here is the secret") is False
```

## Adding New Targets

### 1. Implement RAGTarget Protocol

```python
# apps/my-target/src/my_target/__init__.py
from rag_target import BaseRAGTarget, RAGResponse, RAGContext, RetrievalChunk

class MyRAGTarget(BaseRAGTarget):
    def __init__(self, config):
        super().__init__("my-target", config)
        # Initialize your RAG system
    
    async def query(self, question, **kwargs):
        # Your RAG logic
        return RAGResponse(answer="...", citations=["[1]"])
    
    async def get_context(self, question, **kwargs):
        return RAGContext(query=question, chunks=[...])
    
    async def get_system_prompt(self):
        return "Your system prompt"
    
    async def health_check(self):
        return True
    
    async def close(self):
        # Cleanup
        pass
```

### 2. Add to CLI

```python
# scripts/cli.py
from my_target import MyRAGTarget

@app.command()
def baseline(targets: list[str] = ["reference-rag"]):
    target_instances = []
    for name in targets:
        if name == "my-target":
            target_instances.append(MyRAGTarget(config))
        ...
```

## Adding New Providers

### 1. Implement LLMProvider Protocol

```python
# libs/llm-provider/src/llm_provider/my_provider.py
from llm_provider import BaseLLMProvider, ModelConfig, ProviderType, LLMRequest, LLMResponse

class MyProvider(BaseLLMProvider):
    @property
    def provider_type(self):
        return ProviderType.CUSTOM
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Your completion logic
        pass
    
    async def stream(self, request: LLMRequest):
        # Your streaming logic
        pass
    
    async def health_check(self):
        return True
    
    async def close(self):
        pass
```

### 2. Register in Factory

```python
# libs/llm-provider/src/llm_provider/factory.py
class ProviderFactory:
    def _create_custom(self, config: ModelConfig):
        if config.provider == ProviderType.CUSTOM and config.model_name.startswith("my-"):
            from llm_provider.my_provider import MyProvider
            return MyProvider(config)
        raise NotImplementedError(...)
```

## Testing Guidelines

### Test Categories

| Category | Marker | Description |
|----------|--------|-------------|
| Unit | `@pytest.mark.unit` | Fast, isolated tests |
| Integration | `@pytest.mark.integration` | Requires services |
| Slow | `@pytest.mark.slow` | Long-running tests |
| Requires API | `@pytest.mark.requires_api` | Needs API keys |

### Running Tests

```bash
# Unit tests only (fast)
uv run pytest -m unit

# Integration tests (requires Docker)
uv run pytest -m integration

# All tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=term-missing
```

### Test Data

- Use fixtures for common test data
- Mock external dependencies
- Property-based testing with Hypothesis for edge cases

## Documentation

### Update When

- Adding new features
- Changing CLI interface
- Modifying configuration
- Adding new probes/targets/providers

### Locations

- `README.md` - User-facing documentation
- `docs/` - Detailed guides
- Docstrings - API documentation
- `CHANGELOG.md` - Version history

## Pull Request Process

1. **Branch**: Create from `main` with descriptive name
2. **Develop**: Write code, add tests, update docs
3. **Validate**: Run all checks locally
4. **Submit**: Open PR with clear description
5. **Review**: Address feedback
6. **Merge**: Squash and merge after approval

### PR Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] Type checking passes (`uv run mypy .`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No secrets in code
- [ ] Commit messages follow convention

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. After merge, tag release: `git tag v0.1.0`
5. GitHub Actions builds and publishes

## Community

- **Discussions**: GitHub Discussions for questions
- **Issues**: Bug reports and feature requests
- **Discord**: [Link to community server]
- **Office Hours**: Monthly contributor calls

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, inclusive, and constructive.