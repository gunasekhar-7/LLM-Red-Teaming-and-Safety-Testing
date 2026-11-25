# LLM Red-Teaming & Safety Testing Framework

A production-ready framework for red-teaming RAG (Retrieval-Augmented Generation) applications against prompt injection, data leakage, jailbreaking, and RAG-specific vulnerabilities.

## Features

- **Multi-Target Testing**: Test existing LangChain RAG apps and reference implementations
- **Multi-Provider Support**: OpenAI, Anthropic, Ollama, vLLM, and LiteLLM
- **Comprehensive Probe Suite**: 50+ attack variants across 7 categories
- **Defense-in-Depth**: Input filtering, retrieval sanitation, output sanitization
- **Statistical Validation**: McNemar's test, effect sizes, regression detection
- **Professional Reporting**: HTML/PDF audit reports with executive summaries
- **Interactive Dashboard**: Streamlit-based live testing and visualization
- **Containerized**: Docker Compose for reproducible environments

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
                      Test Harness
├─────────────────────────────────────────────────────────────┤
  Garak Probes + PromptFoo Suites + Custom RAG Probes
├─────────────────────────────────────────────────────────────┤
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Reference    │  │ Existing     │  │ Custom       │
  │ RAG App      │  │ RAG Adapter  │  │ Target       │
  │ (FastAPI)    │  │ (HTTP/LangChain)           │
  └──────────────┘  └──────────────┘  └──────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
         ┌──────────────────────────────────────┐
         │     Multi-Provider LLM Gateway       │
         │  OpenAI │ Anthropic │ Ollama │ vLLM  │
         └──────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐    ┌─────────────┐  ┌───────────┐
    │ Input   │    │ Retrieval   │  │ Output    │
    │ Guard   │    │ Guard       │  │ Guard     │
    └─────────┘    └─────────────┘  └───────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) GPU for local Ollama models

### Installation

```bash
# Clone and setup
git clone <repo>
cd LLM-Red-Teaming

# Install UV (faster than pip)
pip install uv

# Create virtual environment and install
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

### Environment Setup

```bash
# Copy example env and configure
cp .env.example .env

# Edit .env with your API keys
# At minimum, configure one LLM provider
```

### Start Infrastructure

```bash
# Start PostgreSQL, Ollama, Prometheus, Grafana
cd infra/docker
docker compose up -d

# Pull Ollama models (first run)
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

### Initialize Database

```bash
# Initialize pgvector tables
uv run python -m reference_rag.database
# Or via CLI
uv run llm-redteam init-db
```

### Run Baseline Assessment

```bash
# Run against mock targets (no API keys needed)
uv run llm-redteam baseline

# Run with specific providers
uv run llm-redteam baseline --provider ollama-llama31 --provider openai-gpt4o-mini

# Run against your existing RAG (configure adapter first)
uv run llm-redteam baseline --target existing-rag
```

### Run Comparative Test (Post-Mitigation)

```bash
# After implementing mitigations
uv run llm-redteam compare results/baseline/baseline_20241103_120000.jsonl
```

### Generate Audit Report

```bash
# HTML report
uv run llm-redteam report results/baseline/baseline_20241103_120000.jsonl \
  --comparison results/mitigated/comparison.json \
  --format html

# Markdown report
uv run llm-redteam report results/baseline/baseline_20241103_120000.jsonl --format md
```

### Launch Dashboard

```bash
uv run llm-redteam dashboard
# Opens at http://localhost:8501
```

## Project Structure

```
LLM-Red-Teaming/
├── apps/
│   ├── reference-rag/          # Reference RAG implementation (FastAPI + pgvector)
│   └── existing-rag-adapter/   # Adapter for existing LangChain RAG apps
├── libs/
│   ├── rag-target/             # Core RAGTarget protocol
│   ├── llm-provider/           # Multi-provider LLM abstraction
│   ├── guards/                 # Defense layers (input/retrieval/output)
│   └── test-harness/           # Garak + PromptFoo integration
├── infra/
│   └── docker/                 # Docker Compose stack
├── config/
│   └── mitigation_policies.yaml
├── scripts/
│   ├── cli.py                  # Main CLI (llm-redteam)
│   ├── run_baseline.py         # Baseline assessment
│   ├── run_comparative.py      # Comparative testing
│   └── generate_report.py      # Audit report generator
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml              # Workspace configuration
└── .env.example                # Environment template
```

## Attack Categories

| Category | Probes | Description |
|----------|--------|-------------|
| **Prompt Injection** | 8 | Direct/indirect injection, instruction override |
| **Jailbreaking** | 12 | Roleplay, hypothetical, DAN, multi-turn |
| **Data Leakage** | 10 | System prompt, PII, secrets, cross-tenant |
| **RAG-Specific** | 8 | Citation forgery, retrieval hijack, context overflow |
| **Encoding** | 6 | Base64, hex, unicode, high-entropy payloads |
| **Roleplay** | 5 | Persona adoption, authority impersonation |
| **Multi-turn** | 4 | Conversation manipulation, context poisoning |

## Defense Layers

### Input Guard
- **Injection Classifier**: DeBERTa-based semantic detection (threshold: 0.85)
- **Pattern Blocking**: Regex-based known attack patterns
- **Encoding Detection**: Base64, hex, high-entropy detection
- **Length Normalization**: Token limit enforcement (4000 tokens)

### Retrieval Guard
- **PII Redaction**: Microsoft Presidio for entity detection/redaction
- **Relevance Filtering**: Minimum similarity threshold (0.7), max chunks (5)
- **Tenant Isolation**: Strict cross-tenant data separation

### Output Guard
- **Citation Verification**: Validates citations match retrieved sources
- **PII Redaction**: Secondary PII cleanup on responses
- **Refusal Detection**: Identifies proper safety refusals
- **Length Enforcement**: Response token limits (2000 tokens)

## Configuration

### Mitigation Policies (`config/mitigation_policies.yaml`)

All defense layers configurable via YAML:
- Thresholds and sensitivity
- Enabled/disabled components
- Custom patterns and entities
- Per-target overrides

### Target Configuration

Reference RAG configured via environment variables (see `.env.example`):
- LLM provider and model
- Embedding provider and model
- Retrieval parameters (top_k, thresholds)
- Auth settings

## Extending the Framework

### Custom Probes

```python
from test_harness import Probe, AttackCategory, AttackSeverity

class MyCustomProbe(Probe):
    async def generate_prompts(self, context=None):
        return ["Custom attack prompt 1", "Custom attack prompt 2"]
    
    async def evaluate(self, prompt, response, context=None):
        # Return True if defended, False if attack succeeded
        return "refusal" in response.lower()

# Register
harness.add_probe(MyCustomProbe("my_probe", AttackCategory.RAG_SPECIFIC))
```

### Custom Targets

```python
from rag_target import BaseRAGTarget, RAGResponse

class MyRAGTarget(BaseRAGTarget):
    async def query(self, question, **kwargs):
        # Your RAG logic
        return RAGResponse(answer="...", citations=["[1]"])
    
    async def get_context(self, question, **kwargs):
        # Return retrieval context
        pass
    
    async def get_system_prompt(self):
        return "Your system prompt"
    
    async def health_check(self):
        return True
    
    async def close(self):
        pass
```

### Custom Providers

```python
from llm_provider import BaseLLMProvider, ModelConfig, ProviderType

class MyProvider(BaseLLMProvider):
    @property
    def provider_type(self):
        return ProviderType.CUSTOM
    
    async def complete(self, request):
        # Your completion logic
        pass
    
    async def stream(self, request):
        # Your streaming logic
        pass
```

## CI/CD Integration

GitHub Actions workflow (`.github/workflows/redteam.yml`):

```yaml
name: Red-Teaming
on: [push, pull_request]
jobs:
  baseline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - name: Install
        run: pip install uv && uv sync --all-extras
      - name: Start Services
        run: cd infra/docker && docker compose up -d
      - name: Run Baseline
        run: uv run llm-redteam baseline
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with: {name: baseline-results, path: results/baseline/}
```

## Reporting

### HTML Report Includes:
- Executive summary with key metrics
- Scope and methodology
- Target architecture diagram
- Baseline findings ranked by severity
- Mitigations implemented with config
- Comparative validation with statistics
- Residual risk register
- Recommendations roadmap

### Metrics Tracked:
- **Baseline Failure Rate (BFR)**: % attacks succeeding per category
- **Absolute Risk Reduction (ARR)**: BFR_baseline - BFR_mitigated
- **Relative Risk Reduction (RRR)**: ARR / BFR_baseline
- **Number Needed to Treat (NNT)**: Attacks prevented per 100 requests
- **Statistical Significance**: p-value, Cohen's h effect size
- **Regression Detection**: Previously blocked attacks that now succeed

## Security & Ethics

- Attack data never persisted; redacted in logs
- API keys via environment variables only
- Secret scanning in pre-commit hooks
- Responsible disclosure template included
- License compliance tracked via SBOM

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`uv run pytest`)
5. Submit a pull request

## Support

- Issues: GitHub Issues
- Security: See SECURITY.md for responsible disclosure
- Documentation: See `docs/` directory"# LLM-Red-Teaming-and-Safety-Testing" 
