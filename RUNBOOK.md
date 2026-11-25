# LLM Red-Teaming Framework - Operations Runbook

## Overview
This runbook provides operational procedures for the LLM Red-Teaming Framework, including deployment, monitoring, troubleshooting, and incident response.

## Table of Contents
1. [Deployment](#deployment)
2. [Service Management](#service-management)
3. [Monitoring & Alerting](#monitoring--alerting)
4. [Troubleshooting](#troubleshooting)
5. [Incident Response](#incident-response)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Changelog & Version Management](#changelog--version-management)

---

## 1. Deployment

### 1.1 Infrastructure Requirements
- **Docker Engine** v20.10+
- **Docker Compose** v2.20+
- **Minimum 8GB RAM** recommended
- **Minimum 10GB disk space** for results storage

### 1.2 Quick Start (Development)
```bash
# Clone repository
git clone <repository-url>
cd llm-redteaming

# Start infrastructure
cd infra/docker
docker compose up -d

# Wait for services to start
# - PostgreSQL: typically ready in 10s
# - Ollama: typically ready in 30s (first run pulls models)
# - Prometheus/Grafana: typically ready in 15s

# Initialize database
cd ..
uv run llm-redteam init-db

# Run baseline assessment
uv run llm-redteam baseline --target reference-rag --target existing-rag

# Launch dashboard
uv run llm-redteam dashboard
# Dashboard available at: http://localhost:8501
```

### 1.3 Production Deployment (Kubernetes)
```yaml
# Simplified Helm chart values
replicaCount: 3

referenceRag:
  image: ghcr.io/your-org/llm-redteam:latest
  resources:
    limits:
      memory: "512Mi"
      cpu: "500m"
    envFrom:
      - secretRef: llm-redteam-secrets

existingRagAdapter:
  image: ghcr.io/your-org/llm-redteam-adapter:latest
  resources:
    limits:
      memory: "256Mi"
      cpu: "250m"

postgres:
  primaryAffinity: true
  primary: {}

prometheus:
  enabled: true

grafana:
  enabled: true
```

### 1.4 Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `POSTGRES_USER` | Yes | PostgreSQL username |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password |
| `POSTGRES_DB` | Yes | PostgreSQL database name |
| `LLM_PROVIDER` | Yes | One of: openai, anthropic, ollama, litellm |
| `LLM_MODEL` | Yes | Model name (e.g., gpt-4o-mini, claude-3-5-haiku-20241022, llama3.1:8b) |
| `EMBEDDING_PROVIDER` | Yes | Embedding provider |
| `AUTH_ENABLED` | No | Set to "true" for multi-tenant mode |
| `JWT_SECRET` | Yes (if auth enabled) | Cryptographically random 32+ byte secret |
| `OPENAI_API_KEY` | Yes (if openai provider) | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes (if anthropic provider) | Anthropic API key |
| `OLLAMA_BASE_URL` | Yes (if local) | Ollama server URL |
| `LOG_LEVEL` | No | Debug, INFO, WARNING, ERROR |
| `RATE_LIMIT_REQUESTS` | No | Default: 100 |

### 1.5 Service Health Checks
| Service | Endpoint | Expected Response |
|---------|----------|-------------------|
| Reference RAG | `GET /api/v1/health` | `{"status":"healthy","version":"0.1.0","database":true,"rag_service":true}` |
| Existing RAG Mock | `GET /health` | `{"status":"healthy","service":"mock-existing-rag"}` |
| PostgreSQL | `pg_isready` | `PONG` |
| Ollama | `GET /api/tags` | List of available models |
| Prometheus | `GET /-/healthy` | `200 OK` |
| Grafana | `GET /api/health` | `200 OK` |

---

## 2. Service Management

### 2.1 Starting Services
```bash
# From project root
cd infra/docker
docker compose up -d

# Or with specific services
docker compose up -d postgres ollama

# View logs
docker compose logs -f
```

### 2.2 Stopping Services
```bash
cd infra/docker
docker compose down
# To preserve data:
docker compose down -v  # Remove volumes
```

### 2.3 Restarting Services
```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart postgres
# Or: docker compose restart reference-rag
```

### 2.4 Service Dependencies
| Service | Depends On |
|---------|------------|
| reference-rag | postgres, ollama |
| existing-rag-mock | None |
| test-runner | postgres, ollama, reference-rag, existing-rag-mock |
| prometheus | None |
| grafana | prometheus |

---

## 3. Monitoring & Alerting

### 3.1 Prometheus Metrics
The following metrics are exposed at `GET /api/v1/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `rag_requests_total` | Counter | Total requests by method, endpoint, status |
| `rag_request_latency_seconds` | Histogram | Request latency in seconds by method, endpoint |
| `probe_execution_duration_seconds` | Histogram | Per-probe execution duration |
| `guard_latency_seconds` | Histogram | Per-guard execution duration |
| `attack_failure_rate` | Gauge | Real-time failure rate by category |
| `guard_blocked_total` | Counter | Total blocked attacks by guard type |
| `active_connections` | Gauge | Current active connections |

### 3.2 Alerting Rules
```yaml
# Alert if failure rate exceeds 50% for 5 minutes
alert: HighFailureRate
expr: sum by (category) (attack_failure_rate) > 0.5
for: 5m
labels:
  severity: critical
  runbook: "high-failure-rate"

# Alert if p99 latency exceeds 5 seconds
alert: HighLatency
expr: histogram_quantile(0.99, sum(rate(rag_request_latency_seconds_bucket)) by (le) > 5
for: 1m
labels:
  severity: warning
  runbook: "high-latency"

# Alert if new regression detected
alert: NewRegression
expr: increase(guard_new_regressions_total[1h]) > 0
labels:
  severity: critical
  runbook: "regression-detection"
```

### 3.2 Grafana Dashboards
Import the Grafana dashboard provisioning from `infra/docker/grafana/dashboards/`. Key dashboards:

| Dashboard | Purpose |
|-----------|---------|
| Overview | High-level metrics: request rate, failure rate, guard status |
| Attacks | Detailed attack probe results, success/failure rates |
| Guards | Per-guard metrics: latency, block rate, false positives/negatives |
| Services | Service health, connection counts, error rates |
| Regressions | Regression detection, new vs historical |

---

## 4. Troubleshooting

### 4.1 Common Issues

| Symptom | Cause | Resolution |
|---------|--------|------------|
| `docker compose up -d` hangs | Model download in progress | Wait for Ollama first run, or pre-pull models |
| `connection refused` to `/health` | Service not started | `docker compose up -d`, wait 30s |
| `403 Forbidden` on `/query` | Auth not configured | Set `AUTH_ENABLED=true` and configure JWT |
| `database connection failed` | PostgreSQL not ready | `docker compose logs postgres`, wait for startup |
| `rate limit exceeded` | Too many requests | Wait for rate limit reset, configure higher limits |
| `model not found` | Ollama model not pulled | `docker compose exec ollama ollama pull llama3.1:8b` |
| `promptfoo errors` | API rate limits | Set `PROMPTFOO_API_KEY` or use mock targets |
| `OutOfMemory` | Container OOM | Increase resource limits, optimize probe count |
| `timeout on baseline` | Slow LLM or many probes | Reduce `--concurrent`, use `--target` to test specific targets |

### 4.2 Diagnostic Commands

```bash
# Check service health
docker compose ps

# View service logs
docker compose logs <service_name>

# Check PostgreSQL
docker exec <pg-container> pg_isready
docker exec <pg-container> psql -U postgres -d ragdb -c "\dt"

# Check Ollama
docker exec <ollama-container> ollama list
docker exec <ollama-container> ollama ps

# Check Prometheus
curl http://localhost:9090/-/healthy
curl http://localhost:9090/api/v1/targets

# Check Grafana
open http://localhost:3000/api/health

# Test query endpoint
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this service?", "tenant_id": "test"}'
```

### 4.3 Log Analysis
Key log patterns to monitor:
- `ERROR` or `CRITICAL` level logs
- `Request blocked:` messages (guard interventions)
- `Retrieval blocked:` messages
- `Response blocked:` messages
- `Import error` or `Module not found`
- `Rate limit exceeded`
- `Model generation error`
- `Database connection lost`

---

## 5. Incident Response

### 5.1 Incident Severity Levels

| Severity | Criteria | Response Time | Escalation |
|----------|----------|---------------|------------|
| **Critical** | Complete service outage, data leak, 0% defense effectiveness | 15 minutes | Engineering Lead + Security Team |
| **High** | Significant defense degradation (>50% failure rate), persistent errors | 30 minutes | Engineering Team |
| **Medium** | Minor defense degradation, intermittent errors | 2 hours | Engineering Team |
| **Low** | Configuration issues, warnings, non-critical errors | 24 hours | DevOps Team |

### 5.2 Critical Incident Response Procedure

1. **Identify** - Determine the scope and impact
   - Check metrics dashboard
   - Review recent guard logs
   - Verify affected targets and attack vectors

2. **Contain** - Prevent further impact
   - Disable affected probes temporarily
   - Increase guard thresholds if appropriate
   - Restrict API access if needed
   - Rotate API keys if compromised

3. **Eradicate** - Remove the root cause
   - Apply patches/updates
   - Reset secrets/keys
   - Update guard configurations
   - Reload model weights if needed

4. **Recover** - Restore normal operations
   - Restart services
   - Verify all defenses are active
   - Run baseline assessment to confirm improvement
   - Document the incident

5. **Post-Incident Review** - Within 48 hours
   - Root cause analysis
   - Impact assessment
   - Action items and timeline
   - Prevention measures
   - Update runbook and documentation

### 5.3 Escalation Matrix

| Issue Type | First Responder | Escalation | Emergency Contact |
|------------|----------------|------------|-------------------|
| Service outage | SRE on-call | Engineering Lead | pager duty |
| Security breach | Security Team Lead | CTO | Emergency contact |
| Performance degradation | DevOps Engineer | Engineering Manager | Slack #engineering |
| Configuration error | DevOps Engineer | Engineering Lead | Internal ticket |

---

## 6. Maintenance Procedures

### 6.1 Regular Maintenance

| Task | Frequency | Responsible |
|------|-----------|-------------|
| Run baseline assessment | Weekly | DevOps Engineer |
| Update guard configurations | Monthly | Security Engineer |
| Update LLM models | As needed | ML Engineer |
| Review guard effectiveness | Quarterly | Security Team |
| Update threat model | Quarterly | Security Team |
| Update documentation | As needed | Technical Writer |
| Cleanup results older than 90 days | Monthly | DevOps Engineer |

### 6.2 Model Updates

```bash
# Pull new model version
docker compose exec ollama ollama pull llama3.1:8b

# Or specific version
docker compose exec ollama ollama pull llama3.1:8b@sha256:...

# Verify model is available
docker compose exec ollama ollama list

# Restart reference-rag to use new model
docker compose restart reference-rag
```

### 6.3 Dependency Updates

```bash
# Update UV and all packages
uv sync --all-extras --upgrade

# Or update specific packages
uv pip install --upgrade package-name

# Regenerate lockfile
uv lock --upgrade

# Rebuild Docker images
docker compose build
```

### 6.4 Data Retention

- **Baseline results**: Retain for 90 days
- **Mitigation results**: Retain for 1 year
- **Audit reports**: Retain indefinitely
- **Log files**: Rotate weekly, retain for 30 days
- **SBOM files**: Retain for 1 year or until next release

---

## 7. Changelog & Version Management

### 7.1 Version Format
`MAJOR.MINOR.PATCH` (Semantic Versioning 2.0.0)

- **MAJOR**: Breaking API changes, major feature additions
- **MINOR**: New features, non-breaking additions
- **PATCH**: Bug fixes, security updates

### 7.2 Version Bump Procedure

1. **Update version** in `pyproject.toml` (all packages)
2. **Update changelog** in `CHANGELOG.md`
3. **Create Git tag**: `git tag vX.Y.Z`
4. **Push tag**: `git push origin vX.Y.Z`
5. **GitHub Actions** will:
   - Build Docker images
   - Generate SBOMs
   - Create GitHub Release
   - Attach SBOMs to release

### 7.3 Supported Versions

| Version | Release Date | End of Life | Security Updates |
|---------|--------------|-------------|------------------|
| 0.1.x | 2026-11-03 | 2027-11-03 | Critical only |
| 0.0.x | N/A | N/A | N/A |

### 7.4 Downgrade Procedure

```bash
# Pin to specific version
uv pin 0.1.0

# Or set in pyproject.toml
# then reinstall
pip install -e . --force-reinstall
```

---

## Appendix A: Quick Reference Commands

```bash
# Development
cd /path/to/llm-redteaming

# Start infrastructure
cd infra/docker
docker compose up -d

# Initialize database
uv run llm-redteam init-db

# Run assessments
uv run llm-redteam baseline --target reference-rag --target existing-rag --output results/baseline
uv run llm-redteam compare results/baseline/baseline.jsonl --output results/mitigated
uv run llm-redteam report results/baseline/baseline.jsonl --comparison results/mitigated/comparison.json --format html --output results/report.html

# Dashboard
uv run llm-redteam dashboard

# Stop everything
cd infra/docker
docker compose down

# Status
docker compose ps

# Logs
docker compose logs -f reference-rag

# Rebuild images
docker compose build

# Full rebuild from scratch
docker compose down -v
docker compose up -d
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Probe** | An attack vector or test case used to evaluate LLM safety |
| **Guard** | A defensive layer that filters/validates inputs, retrieval, or outputs |
| **Failure Rate** | Percentage of attacks that succeed (bypass all guards) |
| **Regression** | A previously blocked attack that now succeeds after a change |
| **SUT** | System Under Test (the RAG application being evaluated) |
| **SUT** | System Under Test (the RAG application being evaluated) |
| **SBOM** | Software Bill of Materials - lists all dependencies and versions |
| **CI/CD** | Continuous Integration/Continuous Deployment pipeline |
| **SLA** | Service Level Agreement |
| **SLO** | Service Level Objective |
| **SLA** | Service Level Agreement |

---

## Appendix C: Contact Information

- **Security Team**: security@your-organization.com
- **Engineering Lead**: engineering-lead@your-organization.com
- **DevOps On-call**: #engineering-slack (PagerDuty)
- **Documentation**: https://your-doc-site.com/llm-redteam
- **Issue Tracker**: https://your-org.github.io/llm-redteam/issues

---

*This runbook is maintained by the LLM Red-Teaming Team. Last updated: 2026-11-03.*