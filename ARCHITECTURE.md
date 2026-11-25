# Architecture Documentation

## Overview

The LLM Red-Teaming Framework is a modular, extensible system for security testing of RAG applications. It follows a plugin architecture with clear separation of concerns.

## Core Components

### 1. RAG Target Layer (`libs/rag-target`)

**Purpose**: Abstract interface for any testable RAG system.

**Key Abstractions**:
- `RAGTarget` protocol - defines the testable interface
- `BaseRAGTarget` - abstract base class for implementations
- `RAGResponse` - standardized response format
- `RAGContext` - retrieval context with citations
- `RetrievalChunk` - individual document chunk with metadata

**Implementations**:
- `MockExistingRAGAdapter` - for testing without real systems
- `ExistingRAGAdapter` - HTTP/LangChain adapter for production RAG apps
- `ReferenceRAGTarget` - wraps the reference FastAPI RAG app

### 2. LLM Provider Layer (`libs/llm-provider`)

**Purpose**: Unified interface for multiple LLM providers.

**Architecture**:
```
ProviderFactory
    │
    ├── OpenAIProvider → OpenAI API
    ├── OpenAICompatibleProvider → vLLM, local OpenAI-compatible
    ├── AnthropicProvider → Anthropic API
    ├── OllamaProvider → Local Ollama
    └── OllamaEmbeddingProvider → Local embeddings
```

**Key Classes**:
- `ModelConfig` - provider, model, role, parameters
- `LLMRequest` / `LLMResponse` - standardized formats
- `ProviderRegistry` - global provider registry
- `LLMSettings` - environment-based configuration

**Supported Providers**:
| Provider | Models | Use Case |
|----------|--------|----------|
| OpenAI | gpt-4o, gpt-4o-mini | Production, baseline |
| Anthropic | claude-3.5-sonnet, haiku | Comparison, safety |
| Ollama | llama3.1, mistral, nomic-embed | Local, cost-free |
| LiteLLM | Any via LiteLLM proxy | Unified gateway |

### 3. Guards Layer (`libs/guards`)

**Purpose**: Defense-in-depth mitigation layers.

**Pipeline Architecture**:
```
Request → Input Guard → Retrieval Guard → LLM → Output Guard → Response
              │              │                  │
              ▼              ▼                  ▼
        ┌─────────┐    ┌─────────────┐    ┌───────────┐
        │Injection│    │   PII       │    │ Citation  │
        │Classifier│    │ Redaction   │    │ Verify    │
        │Patterns │    │ Relevance   │    │ PII       │
        │Encoding │    │ Tenant Iso  │    │ Refusal   │
        │Length   │    │             │    │ Length    │
        └─────────┘    └─────────────┘    └───────────┘
```

**Guard Categories**:
- **Input Guards**: Pre-LLM protection
- **Retrieval Guards**: Context sanitation
- **Output Guards**: Post-LLM protection
- **Runtime Guards**: Monitoring middleware

**Configuration**: `config/mitigation_policies.yaml`

### 4. Test Harness (`libs/test-harness`)

**Purpose**: Orchestrate attacks, collect results, statistical analysis.

**Core Classes**:
- `TestHarness` - main orchestration
- `Probe` - abstract attack probe
- `AttackResult` - standardized result format
- `ComparativeHarness` - baseline vs mitigated comparison
- `GarakProbeAdapter` - wraps Garak probes
- `PromptFooSuite` - PromptFoo integration

**Probe Types**:
- **Custom Probes**: Python classes for specific attacks
- **Garak Adapters**: Wrap Garak's 100+ built-in probes
- **PromptFoo Cases**: YAML-defined test suites

### 5. Reference RAG (`apps/reference-rag`)

**Purpose**: Production-quality reference implementation.

**Stack**:
- **API**: FastAPI with async SQLAlchemy
- **Vector DB**: PostgreSQL + pgvector
- **Orchestration**: LangChain (retriever + generator)
- **Auth**: JWT + API Key multi-tenancy
- **Observability**: Prometheus metrics, structured logging

**Endpoints**:
| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/api/v1/query` | POST | Yes | RAG query |
| `/api/v1/ingest` | POST | Yes | Document ingestion |
| `/api/v1/documents` | GET | Yes | List documents |
| `/metrics` | GET | No | Prometheus metrics |

### 6. Existing RAG Adapter (`apps/existing-rag-adapter`)

**Purpose**: Test production RAG without modification.

**Modes**:
- **HTTP Mode**: Calls existing REST API
- **Direct Mode**: Uses LangChain runnable in-process

**Configuration**:
```python
ExistingRAGConfig(
    api_base_url="https://your-rag.example.com",
    api_key="your-api-key",
    use_direct=False,  # True for in-process
    langchain_runnable=your_chain,  # For direct mode
)
```

## Data Flow

### Baseline Assessment

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Probes    │────▶│  Test       │────▶│  Targets    │
│  (Attacks)  │     │  Harness    │     │  (RAG Apps) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Providers  │     │  Responses  │
                    │  (LLMs)     │     │  (Evidence) │
                    └─────────────┘     └─────────────┘
                           │                    │
                           └─────────┬──────────┘
                                     ▼
                            ┌─────────────────┐
                            │  Evaluation     │
                            │  (Pass/Fail)    │
                            └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  Results JSONL  │
                            │  + Summary CSV  │
                            │  + HTML Report  │
                            └─────────────────┘
```

### Mitigation Testing

```
Baseline Results ──┐
                   ├──▶ ComparativeHarness ──▶ Statistical Report
Mitigated Results ─┘                          (p-values, effect sizes)
```

## Configuration System

### Layered Configuration

1. **Environment Variables** (`.env`) - Secrets, endpoints
2. **YAML Policies** (`config/mitigation_policies.yaml`) - Defense config
3. **Python Settings** (`pydantic-settings`) - Typed app config
4. **CLI Arguments** - Runtime overrides

### Configuration Precedence

```
CLI Args > Environment > YAML Config > Python Defaults
```

## Extensibility Points

### 1. Custom Probes
```python
class MyProbe(Probe):
    async def generate_prompts(self, context): ...
    async def evaluate(self, prompt, response, context): ...
```

### 2. Custom Targets
```python
class MyTarget(BaseRAGTarget):
    async def query(self, question): ...
    async def get_context(self, question): ...
```

### 3. Custom Providers
```python
class MyProvider(BaseLLMProvider):
    async def complete(self, request): ...
    async def stream(self, request): ...
```

### 4. Custom Guards
```python
class MyGuard(BaseGuard):
    _category = GuardCategory.INPUT  # or RETRIEVAL, OUTPUT
    async def check(self, content, context): ...
```

### 5. Custom Reporters
```python
class MyReporter(Reporter):
    async def generate(self, results): ...
```

## Deployment Architecture

### Development
```
Local Machine
├── Docker Compose (PostgreSQL, Ollama, Prometheus, Grafana)
├── Python venv (framework code)
└── CLI / Dashboard
```

### Production (Containerized)
```
Kubernetes Cluster
├── reference-rag Deployment (HPA, 3+ replicas)
├── existing-rag-adapter Deployment
├── test-runner Job (scheduled/CronJob)
├── PostgreSQL (Cloud SQL / managed)
├── Prometheus + Grafana (monitoring)
└── Object Storage (results, reports)
```

### CI/CD Pipeline
```
Git Push
    │
    ├─▶ Lint & Type Check
    │
    ├─▶ Unit Tests
    │
    ├─▶ Build Docker Images
    │
    ├─▶ Integration Tests (staging)
    │       └─▶ Deploy to staging namespace
    │       └─▶ Run baseline suite
    │
    └─▶ Deploy to Production (on tag)
            └─▶ Scheduled red-teaming runs
```

## Security Boundaries

### Trust Zones

| Zone | Components | Trust Level |
|------|------------|-------------|
| Control Plane | CLI, Dashboard, Orchestration | High |
| Test Execution | Probes, Harness, Providers | Medium |
| Target Systems | RAG Apps, LLMs | Low (under test) |
| Data Storage | Results, Reports, Config | High |

### Data Classification

| Data Type | Classification | Handling |
|-----------|----------------|----------|
| Attack Payloads | CONFIDENTIAL | In-memory only, redacted in logs |
| Successful Exploits | RESTRICTED | Never persisted, immediate redaction |
| Test Results | INTERNAL | Encrypted at rest, access controlled |
| Configuration | INTERNAL | Version controlled, secrets external |

## Performance Considerations

### Concurrency
- Default: 3 concurrent probe executions
- Configurable via `max_concurrent` parameter
- Provider-level rate limiting respected

### Latency Budgets
| Layer | Target p95 | Max p99 |
|-------|------------|---------|
| Input Guard | < 50ms | < 100ms |
| Retrieval Guard | < 20ms | < 50ms |
| Output Guard | < 30ms | < 80ms |
| Total Overhead | < 100ms | < 200ms |

### Scaling
- Horizontal: Multiple test-runner pods
- Vertical: Increase concurrent probes
- Provider: Use multiple API keys for quota

## Monitoring & Observability

### Metrics Collected
- `rag_requests_total` - Request count by target/provider/status
- `rag_request_latency_seconds` - Latency histogram
- `probe_execution_duration` - Per-probe timing
- `guard_latency_seconds` - Per-guard timing
- `attack_failure_rate` - Real-time failure rate

### Alerting Rules
- Failure rate > 50% for 5min → Critical
- p99 latency > 5s → Warning
- New regression detected → Critical
- Provider quota exhausted → Warning

## Future Architecture

### Planned Enhancements
1. **Distributed Tracing** - OpenTelemetry integration
2. **Attack Attribution** - ML-based attack clustering
3. **Adaptive Probes** - Reinforcement learning for novel attacks
4. **Policy as Code** - GitOps for mitigation policies
5. **Multi-Region** - Geo-distributed testing

### Extension Points for v1.0
- Webhook notifications for critical findings
- SIEM integration (Splunk, Elastic, Sentinel)
- Automated PR creation for regression fixes
- API for external tool integration