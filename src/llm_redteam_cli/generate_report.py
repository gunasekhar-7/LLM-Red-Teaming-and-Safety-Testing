"""Professional audit report generator."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from test_harness import AttackResult, ComparativeHarness, AttackCategory, AttackSeverity


class AuditReportGenerator:
    """Generate professional security audit reports."""
    
    def __init__(self, template_dir: Path | None = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        self._ensure_templates()
    
    def _ensure_templates(self):
        """Create default templates if they don't exist."""
        html_template = self.template_dir / "audit_report.html.j2"
        if not html_template.exists():
            self._create_html_template(html_template)
        
        md_template = self.template_dir / "audit_report.md.j2"
        if not md_template.exists():
            self._create_md_template(md_template)
    
    def _create_html_template(self, path: Path):
        """Create default HTML template."""
        template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; 
            color: #1a1a2e; 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 40px 20px;
            background: #fafafa;
        }
        .container { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #1a1a2e; }
        h1 { border-bottom: 3px solid #2563eb; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #e5e7eb; padding-top: 30px; }
        .meta { color: #6b7280; font-size: 0.9em; margin-bottom: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; }
        .metric-value { font-size: 2.5em; font-weight: 700; color: #2563eb; }
        .metric-label { color: #64748b; font-size: 0.9em; }
        .critical { color: #dc2626; }
        .high { color: #ea580c; }
        .medium { color: #ca8a04; }
        .low { color: #16a34a; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        th { background: #f8fafc; font-weight: 600; }
        tr:hover { background: #f8fafc; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; }
        .badge-critical { background: #fef2f2; color: #dc2626; }
        .badge-high { background: #fff7ed; color: #ea580c; }
        .badge-medium { background: #fefce8; color: #ca8a04; }
        .badge-low { background: #f0fdf4; color: #16a34a; }
        .badge-info { background: #eff6ff; color: #2563eb; }
        .finding { background: #fafafa; border-left: 4px solid #2563eb; padding: 16px; margin: 16px 0; border-radius: 0 8px 8px 0; }
        .finding.critical { border-color: #dc2626; }
        .finding.high { border-color: #ea580c; }
        .finding.medium { border-color: #ca8a04; }
        .finding-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .finding-title { font-weight: 600; }
        .finding-evidence { background: #1e293b; color: #e2e8f0; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 0.85em; overflow-x: auto; margin-top: 8px; }
        .toc { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 30px; }
        .toc ul { list-style: none; padding-left: 0; }
        .toc li { padding: 4px 0; }
        .toc a { color: #2563eb; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
        .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.85em; }
        .chart-placeholder { background: #f1f5f9; border: 1px dashed #94a3b8; border-radius: 8px; padding: 40px; text-align: center; color: #64748b; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ metadata.title }}</h1>
            <div class="meta">
                <strong>Version:</strong> {{ metadata.version }} | 
                <strong>Date:</strong> {{ metadata.date }} | 
                <strong>Classification:</strong> {{ metadata.classification }} |
                <strong>Targets:</strong> {{ metadata.targets | join(", ") }}
            </div>
        </header>
        
        <nav class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#executive-summary">1. Executive Summary</a></li>
                <li><a href="#scope-methodology">2. Scope & Methodology</a></li>
                <li><a href="#architecture">3. Target Architecture</a></li>
                <li><a href="#baseline-findings">4. Baseline Findings</a></li>
                <li><a href="#mitigations">5. Mitigations Implemented</a></li>
                <li><a href="#validation">6. Validation Results</a></li>
                <li><a href="#residual-risk">7. Residual Risk Register</a></li>
                <li><a href="#recommendations">8. Recommendations</a></li>
                <li><a href="#appendices">9. Appendices</a></li>
            </ul>
        </nav>
        
        <section id="executive-summary">
            <h2>1. Executive Summary</h2>
            <p>{{ executive_summary }}</p>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f"|format(summary.overall_failure_rate * 100) }}%</div>
                    <div class="metric-label">Baseline Failure Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f"|format(validation.overall.mitigated_failure_rate * 100) if validation else 'N/A' }}</div>
                    <div class="metric-label">Post-Mitigation Failure Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {% if validation and validation.overall.relative_risk_reduction > 0.7 %}critical{% elif validation and validation.overall.relative_risk_reduction > 0.4 %}high{% else %}medium{% endif %}">{{ "%.1f"|format((validation.overall.relative_risk_reduction * 100) if validation else 0) }}%</div>
                    <div class="metric-label">Risk Reduction</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ summary.total_attacks }}</div>
                    <div class="metric-label">Total Tests Executed</div>
                </div>
            </div>
        </section>
        
        <section id="scope-methodology">
            <h2>2. Scope & Methodology</h2>
            <h3>2.1 Scope</h3>
            <p>{{ scope.description }}</p>
            <ul>
                <li><strong>Targets:</strong> {{ scope.targets | join(", ") }}</li>
                <li><strong>Providers:</strong> {{ scope.providers | join(", ") }}</li>
                <li><strong>Attack Categories:</strong> {{ scope.categories | join(", ") }}</li>
                <li><strong>Test Duration:</strong> {{ scope.duration }}</li>
            </ul>
            
            <h3>2.2 Methodology</h3>
            <p>{{ methodology }}</p>
            <ul>
                <li>Automated probe execution via Garak and custom probes</li>
                <li>PromptFoo test suites for structured validation</li>
                <li>Multi-provider testing (OpenAI, Anthropic, Ollama)</li>
                <li>Statistical comparison using McNemar's test and Cohen's h</li>
                <li>Manual verification of critical findings</li>
            </ul>
        </section>
        
        <section id="architecture">
            <h2>3. Target Architecture</h2>
            <p>{{ architecture.description }}</p>
            <div class="chart-placeholder">[Architecture Diagram - Insert Mermaid/PlantUML Diagram]</div>
            <h3>3.1 Component Overview</h3>
            <table>
                <thead><tr><th>Component</th><th>Technology</th><th>Version</th><th>Configuration</th></tr></thead>
                <tbody>
                {% for comp in architecture.components %}
                    <tr><td>{{ comp.name }}</td><td>{{ comp.tech }}</td><td>{{ comp.version }}</td><td>{{ comp.config }}</td></tr>
                {% endfor %}
                </tbody>
            </table>
        </section>
        
        <section id="baseline-findings">
            <h2>4. Baseline Findings</h2>
            <p>Initial assessment executed {{ summary.total_attacks }} attack attempts across {{ summary.by_category | length }} categories.</p>
            
            <h3>4.1 Failure Rate by Category</h3>
            <table>
                <thead><tr><th>Category</th><th>Tests</th><th>Failures</th><th>Failure Rate</th><th>Severity</th></tr></thead>
                <tbody>
                {% for cat, rate in summary.by_category.mean.items() %}
                    <tr>
                        <td>{{ cat.replace('_', ' ').title() }}</td>
                        <td>{{ summary.by_category.count[cat] }}</td>
                        <td>{{ (summary.by_category.count[cat] * rate) | int }}</td>
                        <td class="{% if rate > 0.5 %}critical{% elif rate > 0.2 %}high{% elif rate > 0.1 %}medium{% else %}low{% endif %}">{{ "%.1f"|format(rate * 100) }}%</td>
                        <td>{{ severity_for_category(cat) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            
            <h3>4.2 Top Critical Findings</h3>
            {% for finding in top_findings[:10] %}
            <div class="finding {{ finding.severity.lower() }}">
                <div class="finding-header">
                    <span class="finding-title">{{ finding.probe_name }}</span>
                    <span class="badge badge-{{ finding.severity.lower() }}">{{ finding.severity }}</span>
                </div>
                <p><strong>Target:</strong> {{ finding.target_name }} | <strong>Category:</strong> {{ finding.category }}</p>
                <p><strong>Attack:</strong> {{ finding.prompt[:200] }}{% if finding.prompt|length > 200 %}...{% endif %}</p>
                <p><strong>Response:</strong></p>
                <div class="finding-evidence">{{ finding.response[:500] }}{% if finding.response|length > 500 %}...{% endif %}</div>
            </div>
            {% endfor %}
        </section>
        
        <section id="mitigations">
            <h2>5. Mitigations Implemented</h2>
            <p>{{ mitigations.overview }}</p>
            
            <h3>5.1 Defense Layers</h3>
            <table>
                <thead><tr><th>Layer</th><th>Components</th><th>Coverage</th><th>Latency Overhead</th></tr></thead>
                <tbody>
                {% for layer in mitigations.layers %}
                    <tr>
                        <td>{{ layer.name }}</td>
                        <td>{{ layer.components | join(", ") }}</td>
                        <td>{{ layer.coverage }}</td>
                        <td>{{ layer.latency }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            
            <h3>5.2 Configuration Highlights</h3>
            <pre>{{ mitigations.config_yaml }}</pre>
        </section>
        
        <section id="validation">
            <h2>6. Validation Results</h2>
            {% if validation %}
            <h3>6.1 Overall Comparison</h3>
            <table>
                <thead><tr><th>Metric</th><th>Baseline</th><th>Mitigated</th><th>Change</th></tr></thead>
                <tbody>
                    <tr><td>Failure Rate</td><td>{{ "%.1f"|format(validation.overall.baseline_failure_rate * 100) }}%</td><td>{{ "%.1f"|format(validation.overall.mitigated_failure_rate * 100) }}%</td><td class="{% if validation.overall.absolute_risk_reduction > 0 %}low{% else %}critical{% endif %}">{{ "%+.1f"|format(validation.overall.absolute_risk_reduction * 100) }}%</td></tr>
                    <tr><td>Relative Reduction</td><td>-</td><td>-</td><td>{{ "%.1f"|format(validation.overall.relative_risk_reduction * 100) }}%</td></tr>
                    <tr><td>Statistical Significance</td><td colspan="3">p = {{ "%.6f"|format(validation.overall.p_value) }} {% if validation.overall.significant %}(Significant){% else %}(Not Significant){% endif %}</td></tr>
                    <tr><td>Effect Size (Cohen's h)</td><td colspan="3">{{ "%.3f"|format(validation.overall.effect_size) }} {% if validation.overall.effect_size > 0.8 %}(Large){% elif validation.overall.effect_size > 0.5 %}(Medium){% else %}(Small){% endif %}</td></tr>
                </tbody>
            </table>
            
            <h3>6.2 Category-Level Results</h3>
            <table>
                <thead><tr><th>Category</th><th>Baseline</th><th>Mitigated</th><th>Abs. Reduction</th><th>Rel. Reduction</th></tr></thead>
                <tbody>
                {% for cat, data in validation.by_category.items() %}
                    <tr>
                        <td>{{ cat.replace('_', ' ').title() }}</td>
                        <td>{{ "%.1f"|format(data.baseline_failure_rate * 100) }}%</td>
                        <td>{{ "%.1f"|format(data.mitigated_failure_rate * 100) }}%</td>
                        <td class="{% if data.absolute_reduction > 0 %}low{% else %}critical{% endif %}">{{ "%+.1f"|format(data.absolute_reduction * 100) }}%</td>
                        <td>{{ "%.1f"|format(data.relative_reduction * 100) }}%</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            
            {% if validation.regressions %}
            <h3>6.3 Regressions Detected</h3>
            <p class="critical">⚠️ {{ validation.regressions | length }} regression(s) found - previously blocked attacks now succeed.</p>
            <table>
                <thead><tr><th>Probe</th><th>Target</th><th>Category</th><th>Severity</th></tr></thead>
                <tbody>
                {% for reg in validation.regressions %}
                    <tr>
                        <td>{{ reg.probe }}</td>
                        <td>{{ reg.target }}</td>
                        <td>{{ reg.category }}</td>
                        <td><span class="badge badge-{{ reg.severity.lower() }}">{{ reg.severity }}</span></td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            {% else %}
            <h3>6.3 Regressions</h3>
            <p class="low">✓ No regressions detected - all previously blocked attacks remain blocked.</p>
            {% endif %}
            {% else %}
            <p>Validation results not available. Run comparative testing to generate.</p>
            {% endif %}
        </section>
        
        <section id="residual-risk">
            <h2>7. Residual Risk Register</h2>
            <p>After mitigations, the following risks remain:</p>
            <table>
                <thead><tr><th>Risk ID</th><th>Category</th><th>Residual Likelihood</th><th>Impact</th><th>Risk Score</th><th>Status</th></tr></thead>
                <tbody>
                {% for risk in residual_risks %}
                    <tr>
                        <td>{{ risk.id }}</td>
                        <td>{{ risk.category }}</td>
                        <td>{{ risk.likelihood }}</td>
                        <td>{{ risk.impact }}</td>
                        <td class="{{ risk.score_class }}">{{ risk.score }}</td>
                        <td>{{ risk.status }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </section>
        
        <section id="recommendations">
            <h2>8. Recommendations</h2>
            <ol>
            {% for i, rec in enumerate(recommendations, 1) %}
                <li><strong>{{ rec.title }}</strong>: {{ rec.description }}</li>
            {% endfor %}
            </ol>
        </section>
        
        <section id="appendices">
            <h2>9. Appendices</h2>
            <h3>9.1 Probe Catalog</h3>
            <table>
                <thead><tr><th>Probe</th><th>Category</th><th>Severity</th><th>Description</th></tr></thead>
                <tbody>
                {% for probe in probe_catalog %}
                    <tr>
                        <td>{{ probe.name }}</td>
                        <td>{{ probe.category }}</td>
                        <td><span class="badge badge-{{ probe.severity.lower() }}">{{ probe.severity }}</span></td>
                        <td>{{ probe.description }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            
            <h3>9.2 Full Test Results</h3>
            <p>Complete results available in accompanying JSONL/CSV files.</p>
            
            <h3>9.3 Configuration</h3>
            <pre>{{ full_config }}</pre>
        </section>
        
        <footer class="footer">
            <p>Report generated by LLM Red-Teaming Framework v{{ metadata.version }}</p>
            <p>{{ metadata.date }} | Classification: {{ metadata.classification }}</p>
        </footer>
    </div>
</body>
</html>"""
        path.write_text(template, encoding='utf-8')
    
    def _create_md_template(self, path: Path):
        """Create default Markdown template."""
        template = """# {{ metadata.title }}

**Version:** {{ metadata.version }} | **Date:** {{ metadata.date }} | **Classification:** {{ metadata.classification }}

---

## 1. Executive Summary

{{ executive_summary }}

| Metric | Value |
|--------|-------|
| Baseline Failure Rate | {{ "%.1f"|format(summary.overall_failure_rate * 100) }}% |
| Post-Mitigation Failure Rate | {{ "%.1f"|format(validation.overall.mitigated_failure_rate * 100) if validation else 'N/A' }}% |
| Risk Reduction | {{ "%.1f"|format((validation.overall.relative_risk_reduction * 100) if validation else 0) }}% |
| Total Tests | {{ summary.total_attacks }} |

---

## 2. Scope & Methodology

### 2.1 Scope

{{ scope.description }}

- **Targets:** {{ scope.targets | join(", ") }}
- **Providers:** {{ scope.providers | join(", ") }}
- **Attack Categories:** {{ scope.categories | join(", ") }}
- **Test Duration:** {{ scope.duration }}

### 2.2 Methodology

{{ methodology }}

---

## 3. Target Architecture

{{ architecture.description }}

| Component | Technology | Version | Configuration |
|-----------|------------|---------|---------------|
{% for comp in architecture.components %}
| {{ comp.name }} | {{ comp.tech }} | {{ comp.version }} | {{ comp.config }} |
{% endfor %}

---

## 4. Baseline Findings

### 4.1 Failure Rate by Category

| Category | Tests | Failures | Failure Rate | Severity |
|----------|-------|----------|--------------|----------|
{% for cat, rate in summary.by_category.mean.items() %}
| {{ cat.replace('_', ' ').title() }} | {{ summary.by_category.count[cat] }} | {{ (summary.by_category.count[cat] * rate) | int }} | {{ "%.1f"|format(rate * 100) }}% | {{ severity_for_category(cat) }} |
{% endfor %}

### 4.2 Top Critical Findings

{% for finding in top_findings[:10] %}
#### {{ finding.probe_name }} ({{ finding.severity }})

- **Target:** {{ finding.target_name }}
- **Category:** {{ finding.category }}
- **Attack:** {{ finding.prompt[:200] }}{% if finding.prompt|length > 200 %}...{% endif %}
- **Response:** `{{ finding.response[:200] }}{% if finding.response|length > 200 %}...{% endif %}`

---
{% endfor %}

---

## 5. Mitigations Implemented

{{ mitigations.overview }}

### 5.1 Defense Layers

| Layer | Components | Coverage | Latency Overhead |
|-------|------------|----------|------------------|
{% for layer in mitigations.layers %}
| {{ layer.name }} | {{ layer.components | join(", ") }} | {{ layer.coverage }} | {{ layer.latency }} |
{% endfor %}

### 5.2 Configuration

```yaml
{{ mitigations.config_yaml }}
```

---

## 6. Validation Results

{% if validation %}
### 6.1 Overall Comparison

| Metric | Baseline | Mitigated | Change |
|--------|----------|-----------|--------|
| Failure Rate | {{ "%.1f"|format(validation.overall.baseline_failure_rate * 100) }}% | {{ "%.1f"|format(validation.overall.mitigated_failure_rate * 100) }}% | {{ "%+.1f"|format(validation.overall.absolute_risk_reduction * 100) }}% |
| Relative Reduction | - | - | {{ "%.1f"|format(validation.overall.relative_risk_reduction * 100) }}% |
| P-value | - | - | {{ "%.6f"|format(validation.overall.p_value) }} {% if validation.overall.significant %}(Significant){% else %}(Not Significant){% endif %} |
| Effect Size | - | - | {{ "%.3f"|format(validation.overall.effect_size) }} |

### 6.2 Category-Level Results

| Category | Baseline | Mitigated | Abs. Reduction | Rel. Reduction |
|----------|----------|-----------|----------------|----------------|
{% for cat, data in validation.by_category.items() %}
| {{ cat.replace('_', ' ').title() }} | {{ "%.1f"|format(data.baseline_failure_rate * 100) }}% | {{ "%.1f"|format(data.mitigated_failure_rate * 100) }}% | {{ "%+.1f"|format(data.absolute_reduction * 100) }}% | {{ "%.1f"|format(data.relative_reduction * 100) }}% |
{% endfor %}

{% if validation.regressions %}
### 6.3 Regressions Detected

⚠️ {{ validation.regressions | length }} regression(s) found.

| Probe | Target | Category | Severity |
|-------|--------|----------|----------|
{% for reg in validation.regressions %}
| {{ reg.probe }} | {{ reg.target }} | {{ reg.category }} | {{ reg.severity }} |
{% endfor %}
{% else %}
### 6.3 Regressions

✓ No regressions detected.
{% endif %}
{% else %}
Validation results not available.
{% endif %}

---

## 7. Residual Risk Register

| Risk ID | Category | Likelihood | Impact | Score | Status |
|---------|----------|------------|--------|-------|--------|
{% for risk in residual_risks %}
| {{ risk.id }} | {{ risk.category }} | {{ risk.likelihood }} | {{ risk.impact }} | {{ risk.score }} | {{ risk.status }} |
{% endfor %}

---

## 8. Recommendations

{% for i, rec in enumerate(recommendations, 1) %}
{{ i }}. **{{ rec.title }}**: {{ rec.description }}
{% endfor %}

---

## 9. Appendices

### 9.1 Probe Catalog

| Probe | Category | Severity | Description |
|-------|----------|----------|-------------|
{% for probe in probe_catalog %}
| {{ probe.name }} | {{ probe.category }} | {{ probe.severity }} | {{ probe.description }} |
{% endfor %}

### 9.2 Configuration

```yaml
{{ full_config }}
```
"""
        path.write_text(template, encoding='utf-8')
    
    def generate(
        self,
        baseline_results: list[AttackResult],
        comparison: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        output_path: Path | None = None,
        format: str = "html",
    ) -> Path:
        """Generate audit report."""
        # Compute summary
        from test_harness import TestHarness
        harness = TestHarness(targets=[], probes=[])
        harness.results = baseline_results
        summary = harness.generate_summary()
        
        # Prepare template data
        template_data = {
            "metadata": metadata or {
                "title": "LLM Red-Teaming Security Audit Report",
                "version": "1.0.0",
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "classification": "CONFIDENTIAL",
                "targets": list(set(r.target_name for r in baseline_results)),
            },
            "summary": summary,
            "validation": comparison,
            "scope": {
                "description": "Security assessment of RAG-based LLM applications against prompt injection, data leakage, jailbreaking, and RAG-specific vulnerabilities.",
                "targets": list(set(r.target_name for r in baseline_results)),
                "providers": list(set(r.provider_name for r in baseline_results)),
                "categories": list(set(r.category.value if hasattr(r.category, 'value') else r.category for r in baseline_results)),
                "duration": "Automated",
            },
            "methodology": "Automated red-teaming using Garak probes, PromptFoo test suites, and custom RAG-specific attack vectors. Each probe executed against all target-provider combinations. Results evaluated for attack success (failure to defend) vs defense (block/refusal). Statistical validation via chi-squared test and effect size measurement.",
            "architecture": {
                "description": "Multi-tenant RAG application with LangChain orchestration, pgvector for embeddings, and configurable LLM providers.",
                "components": [
                    {"name": "API Gateway", "tech": "FastAPI", "version": "0.109", "config": "JWT/API Key auth, rate limiting"},
                    {"name": "RAG Orchestrator", "tech": "LangChain", "version": "0.1.x", "config": "Custom chain with retriever + generator"},
                    {"name": "Vector Store", "tech": "PostgreSQL + pgvector", "version": "16/0.5", "config": "IVFFlat index, cosine similarity"},
                    {"name": "LLM Providers", "tech": "OpenAI/Anthropic/Ollama", "version": "Various", "config": "gpt-4o-mini, claude-3.5-haiku, llama3.1:8b"},
                    {"name": "Embeddings", "tech": "OpenAI/Ollama", "version": "Various", "config": "text-embedding-3-small, nomic-embed-text"},
                ],
            },
            "top_findings": sorted(
                [r for r in baseline_results if not r.passed],
                key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(x.severity.value if hasattr(x.severity, 'value') else x.severity, 5)
            ),
            "mitigations": {
                "overview": "Three-layer defense-in-depth: Input filtering (injection classifier, pattern blocking, encoding detection), Retrieval sanitation (PII redaction, relevance filtering, tenant isolation), Output sanitization (citation verification, PII redaction, refusal detection).",
                "layers": [
                    {"name": "Input Guard", "components": ["Injection Classifier (DeBERTa)", "Pattern Blocking", "Encoding Detection", "Length Normalization"], "coverage": "Prompt Injection, Encoding", "latency": "< 50ms p95"},
                    {"name": "Retrieval Guard", "components": ["PII Redaction (Presidio)", "Relevance Filtering", "Tenant Isolation"], "coverage": "Data Leakage, Cross-tenant", "latency": "< 20ms p95"},
                    {"name": "Output Guard", "components": ["Citation Verification", "PII Redaction", "Refusal Detection", "Length Enforcement"], "coverage": "Citation Forgery, Data Leakage", "latency": "< 30ms p95"},
                ],
                "config_yaml": """input_guard:
  injection_classifier_threshold: 0.85
  max_prompt_tokens: 4000
  blocked_patterns: ["ignore previous", "system prompt", "### Instruction"]
  encoding_detection: true

retrieval_guard:
  pii_redaction: true
  min_relevance_score: 0.7
  max_chunks: 5
  tenant_isolation: strict

output_guard:
  pii_redaction: true
  citation_verification: true
  refusal_detection: true
  max_response_tokens: 2000""",
            },
            "residual_risks": [
                {"id": "RISK-001", "category": "Prompt Injection", "likelihood": "Low", "impact": "High", "score": "Medium", "score_class": "medium", "status": "Monitored"},
                {"id": "RISK-002", "category": "Novel Jailbreaks", "likelihood": "Medium", "impact": "High", "score": "Medium", "score_class": "medium", "status": "Continuous Testing"},
                {"id": "RISK-003", "category": "Multi-turn Attacks", "likelihood": "Medium", "impact": "Medium", "score": "Medium", "score_class": "medium", "status": "Partial Coverage"},
                {"id": "RISK-004", "category": "Encoding Variants", "likelihood": "Low", "impact": "Medium", "score": "Low", "score_class": "low", "status": "Monitored"},
            ],
            "recommendations": [
                {"title": "Continuous Red-Teaming", "description": "Integrate automated testing into CI/CD pipeline for regression detection on every deployment."},
                {"title": "Expand Probe Coverage", "description": "Add multi-turn conversation probes, token smuggling attacks, and model-specific bypasses."},
                {"title": "Adaptive Defense", "description": "Implement learning-based detection that adapts to novel attack patterns observed in production."},
                {"title": "Enhanced Monitoring", "description": "Deploy real-time anomaly detection on production traffic with automated quarantine."},
                {"title": "Regular Re-assessment", "description": "Schedule quarterly comprehensive re-testing against updated threat landscape."},
            ],
            "probe_catalog": [
                {"name": p.name, "category": p.category.value, "severity": p.severity.value, "description": p.description}
                for p in create_baseline_probes()
            ],
            "full_config": """# Full configuration would be inserted here
# See config/mitigation_policies.yaml""",
        }
        
        # Add helper functions
        template_data["severity_for_category"] = lambda cat: {
            "prompt_injection": "High",
            "jailbreak": "High",
            "data_leakage": "Critical",
            "rag_specific": "High",
            "encoding": "Medium",
            "roleplay": "Medium",
            "multiturn": "Medium",
        }.get(cat, "Medium")
        
        # Render template
        template_name = f"audit_report.{format}.j2"
        template = self.env.get_template(template_name)
        rendered = template.render(**template_data)
        
        # Save
        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"audit_report_{timestamp}.{format}")
        
        output_path.write_text(rendered, encoding='utf-8')
        return output_path


def create_baseline_probes():
    """Import probe creation to avoid circular imports."""
    from .run_baseline import create_baseline_probes as _create
    return _create()


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate audit report")
    parser.add_argument("results", type=Path, help="Baseline results JSONL")
    parser.add_argument("--comparison", type=Path, help="Comparison JSON")
    parser.add_argument("--format", choices=["html", "md"], default="html")
    parser.add_argument("--output", type=Path, help="Output file path")
    from datetime import datetime

    args = parser.parse_args()
    
    # Load results
    results = []
    with open(args.results) as f:
        for line in f:
            data = json.loads(line)
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            data['category'] = AttackCategory(data['category'])
            data['severity'] = AttackSeverity(data['severity'])
            results.append(AttackResult(**data))
    
    # Load comparison if provided
    comparison = None
    if args.comparison and args.comparison.exists():
        with open(args.comparison) as f:
            comparison = json.load(f)
    
    # Generate report
    generator = AuditReportGenerator()
    output = generator.generate(
        baseline_results=results,
        comparison=comparison,
        output_path=args.output,
        format=args.format,
    )
    
    print(f"Report generated: {output}")


if __name__ == "__main__":
    asyncio.run(main())