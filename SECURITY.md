# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow responsible disclosure:

### How to Report

1. **Do NOT** create a public GitHub issue
2. Email security@llm-redteam.example.com with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)
3. We will acknowledge within 48 hours
4. We aim to patch critical issues within 7 days

### What to Include

- Affected component(s)
- Attack vector description
- Proof of concept (if safe to share)
- Environment details (OS, Python version, dependencies)
- Any mitigations you've identified

## Security Features

### Built-in Protections

- **Input Validation**: All user inputs sanitized and validated
- **Secret Management**: API keys only via environment variables
- **Audit Logging**: All attack attempts logged (redacted)
- **Rate Limiting**: Configurable per-endpoint limits
- **Authentication**: JWT/API Key support for multi-tenancy

### Data Handling

- Attack payloads never persisted to disk
- PII automatically redacted from logs
- Results files contain only metadata, not raw exploits
- SBOM generated for supply chain transparency

## Threat Model

### In Scope

- Prompt injection via user queries
- Data leakage through retrieval
- Jailbreaking via roleplay/encoding
- RAG-specific manipulation
- Cross-tenant data access

### Out of Scope

- Infrastructure attacks (DDoS, etc.)
- Supply chain attacks on dependencies
- Physical security
- Social engineering

## Secure Development

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Runs on every commit:
# - ruff (linting)
# - mypy (type checking)
# - detect-secrets (secret scanning)
# - pytest (unit tests)
```

### Dependency Scanning

```bash
# Generate SBOM
uv run syft packages . -o spdx-json > sbom.spdx.json

# Check for vulnerabilities
uv run pip-audit
```

## Configuration Security

### Required Settings

- `JWT_SECRET`: Must be cryptographically random (32+ bytes)
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`: Rotate regularly
- `DATABASE_URL`: Use strong passwords, restrict network access

### Recommended Hardening

```yaml
# mitigation_policies.yaml
input_guard:
  injection_classifier_threshold: 0.9  # Higher = more sensitive
  
retrieval_guard:
  tenant_isolation: strict  # Never use 'none' in production
  
output_guard:
  citation_verification_enabled: true
```

## Incident Response

### If Compromise Suspected

1. **Isolate**: Stop affected services
2. **Assess**: Review logs for attack patterns
3. **Contain**: Block malicious IPs/tenants
4. **Eradicate**: Deploy patches, rotate secrets
5. **Recover**: Restore from clean backups
6. **Learn**: Update probes, improve defenses

### Log Analysis

Key log patterns to monitor:
- High failure rates on specific probes
- Repeated blocked patterns from same source
- Anomalous latency spikes
- Cross-tenant access attempts

## Compliance

This framework supports compliance with:
- **OWASP LLM Top 10** coverage
- **NIST AI RMF** mapping
- **ISO 42001** alignment
- **EU AI Act** preparation

## Contact

Security Team: security@llm-redteam.example.com
PGP Key: Available on request