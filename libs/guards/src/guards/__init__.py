"""Defensive guard layers for LLM safety."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GuardAction(str, Enum):
    """Action taken by a guard."""
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    MODIFY = "modify"
    QUARANTINE = "quarantine"


class GuardCategory(str, Enum):
    """Category of guard."""
    INPUT = "input"
    RETRIEVAL = "retrieval"
    OUTPUT = "output"
    RUNTIME = "runtime"


@dataclass
class GuardResult:
    """Result of a guard check."""
    action: GuardAction
    category: GuardCategory
    guard_name: str
    message: str
    modified_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class GuardConfig(BaseModel):
    """Configuration for a guard."""
    enabled: bool = True
    priority: int = 100  # Lower runs first
    threshold: float = 0.8
    params: dict[str, Any] = Field(default_factory=dict)


class GuardSettings(BaseSettings):
    """Global guard settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Injection classifier
    injection_classifier_model: str = "protectai/deberta-v3-base-prompt-injection"
    injection_threshold: float = 0.85
    
    # PII
    pii_enabled: bool = True
    pii_threshold: float = 0.7
    pii_entities: list[str] = Field(default_factory=lambda: [
        "PERSON", "EMAIL", "PHONE_NUMBER", "CREDIT_CARD", "IBAN",
        "IP_ADDRESS", "API_KEY", "PASSWORD", "SECRET_KEY"
    ])
    
    # Patterns
    blocked_patterns: list[str] = Field(default_factory=lambda: [
        r"ignore previous instructions",
        r"system prompt",
        r"### Instruction",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[\/INST\]",
        r"\[INST\]",
    ])
    
    # Encoding detection
    encoding_detection_enabled: bool = True
    
    # Retrieval
    min_relevance_score: float = 0.7
    max_chunks: int = 5
    tenant_isolation: str = "strict"  # strict, warn, none
    
    # Output
    citation_verification_enabled: bool = True
    refusal_detection_enabled: bool = True
    max_response_tokens: int = 2000


class BaseGuard(ABC):
    """Abstract base class for all guards."""
    
    def __init__(self, name: str, config: GuardConfig, settings: GuardSettings | None = None):
        self.name = name
        self.config = config
        self.settings = settings or GuardSettings()
        self._enabled = config.enabled
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @property
    def category(self) -> GuardCategory:
        return self._category
    
    @property
    @abstractmethod
    def _category(self) -> GuardCategory:
        pass
    
    @abstractmethod
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        pass
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False


class GuardPipeline:
    """Pipeline for running multiple guards in sequence."""
    
    def __init__(self, guards: list[BaseGuard] | None = None):
        self.guards = guards or []
        # Sort by priority
        self.guards.sort(key=lambda g: g.config.priority)
    
    def add_guard(self, guard: BaseGuard) -> None:
        self.guards.append(guard)
        self.guards.sort(key=lambda g: g.config.priority)
    
    def remove_guard(self, name: str) -> bool:
        for i, guard in enumerate(self.guards):
            if guard.name == name:
                self.guards.pop(i)
                return True
        return False
    
    async def run(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        stop_on_block: bool = True,
    ) -> tuple[str, list[GuardResult]]:
        """Run all guards on content."""
        results = []
        current_content = content
        
        for guard in self.guards:
            if not guard.enabled:
                continue
            
            result = await guard.check(current_content, context)
            results.append(result)
            
            if result.action == GuardAction.BLOCK and stop_on_block:
                return current_content, results
            
            if result.action == GuardAction.MODIFY and result.modified_content:
                current_content = result.modified_content
            
            if result.action == GuardAction.REDACT and result.modified_content:
                current_content = result.modified_content
        
        return current_content, results


# Input Guards
class InjectionClassifierGuard(BaseGuard):
    """Semantic prompt injection classifier using DeBERTa."""
    
    _category = GuardCategory.INPUT
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("injection_classifier", config, settings)
        self._classifier = None
    
    async def _get_classifier(self):
        if self._classifier is None:
            try:
                from transformers import pipeline
                self._classifier = pipeline(
                    "text-classification",
                    model=self.settings.injection_classifier_model,
                    device=-1,  # CPU
                )
            except Exception:
                # Fallback to rule-based if transformers not available
                self._classifier = "rule_based"
        return self._classifier
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        classifier = await self._get_classifier()
        
        if classifier == "rule_based":
            return await self._rule_based_check(content)
        
        try:
            result = classifier(content)[0]
            score = result["score"] if result["label"] == "INJECTION" else 1 - result["score"]
            
            if score >= self.config.threshold:
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category=self._category,
                    guard_name=self.name,
                    message=f"Prompt injection detected (score: {score:.3f})",
                    confidence=score,
                )
            
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="No injection detected",
                confidence=1 - score,
            )
        except Exception as e:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message=f"Classifier error, allowing: {e}",
                confidence=0.0,
            )
    
    async def _rule_based_check(self, content: str) -> GuardResult:
        """Fallback rule-based detection."""
        injection_patterns = [
            "ignore previous",
            "system prompt",
            "you are now",
            "pretend to be",
            "roleplay as",
            "act as",
            "disregard",
            "override",
            "bypass",
            "jailbreak",
        ]
        
        content_lower = content.lower()
        for pattern in injection_patterns:
            if pattern in content_lower:
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category=self._category,
                    guard_name=self.name,
                    message=f"Rule-based injection pattern detected: {pattern}",
                    confidence=0.7,
                )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message="No rule-based injection patterns detected",
            confidence=0.9,
        )


class PatternBlockGuard(BaseGuard):
    """Block/redact based on regex patterns."""
    
    _category = GuardCategory.INPUT
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("pattern_block", config, settings)
        import regex as re
        patterns = config.params.get("patterns", self.settings.blocked_patterns)
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        for pattern in self._compiled:
            if pattern.search(content):
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category=self._category,
                    guard_name=self.name,
                    message=f"Blocked pattern matched: {pattern.pattern}",
                    confidence=0.9,
                )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message="No blocked patterns found",
            confidence=1.0,
        )


class EncodingDetectionGuard(BaseGuard):
    """Detect encoded payloads (base64, hex, unicode, etc.)."""
    
    _category = GuardCategory.INPUT
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("encoding_detection", config, settings)
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.encoding_detection_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Encoding detection disabled",
                confidence=1.0,
            )
        
        # Check for high entropy (potential encoding)
        import math
        from collections import Counter
        
        # Base64 detection
        import base64
        import binascii
        
        def is_base64(s: str) -> bool:
            try:
                if len(s) % 4 == 0:
                    base64.b64decode(s, validate=True)
                    return True
            except Exception:
                pass
            return False
        
        def is_hex(s: str) -> bool:
            try:
                binascii.unhexlify(s)
                return True
            except Exception:
                return False
        
        # Check words for encoding
        words = content.split()
        for word in words:
            if len(word) > 20 and (is_base64(word) or is_hex(word)):
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category=self._category,
                    guard_name=self.name,
                    message=f"Encoded payload detected: {word[:50]}...",
                    confidence=0.8,
                )
        
        # Entropy check
        if len(content) > 100:
            counter = Counter(content)
            entropy = -sum(
                (count / len(content)) * math.log2(count / len(content))
                for count in counter.values()
            )
            if entropy > 4.5:  # High entropy threshold
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category=self._category,
                    guard_name=self.name,
                    message=f"High entropy content detected (entropy: {entropy:.2f})",
                    confidence=0.6,
                )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message="No encoding detected",
            confidence=0.9,
        )


class LengthNormalizationGuard(BaseGuard):
    """Normalize and validate input length."""
    
    _category = GuardCategory.INPUT
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("length_normalization", config, settings)
        self.max_tokens = config.params.get("max_tokens", 4000)
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        # Rough token estimation (4 chars per token)
        estimated_tokens = len(content) / 4
        
        if estimated_tokens > self.max_tokens:
            return GuardResult(
                action=GuardAction.BLOCK,
                category=self._category,
                guard_name=self.name,
                message=f"Input too long: ~{estimated_tokens:.0f} tokens (max: {self.max_tokens})",
                confidence=1.0,
            )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message=f"Input length OK: ~{estimated_tokens:.0f} tokens",
            confidence=1.0,
        )


# Retrieval Guards
class PIIRedactionGuard(BaseGuard):
    """Redact PII from retrieved chunks."""
    
    _category = GuardCategory.RETRIEVAL
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("pii_redaction", config, settings)
        self._analyzer = None
        self._anonymizer = None
    
    async def _get_analyzer(self):
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except Exception:
                self._analyzer = "disabled"
        return self._analyzer
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.pii_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="PII redaction disabled",
                confidence=1.0,
            )
        
        analyzer = await self._get_analyzer()
        if analyzer == "disabled":
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Presidio not available, skipping PII redaction",
                confidence=0.5,
            )
        
        try:
            results = analyzer.analyze(
                text=content,
                entities=self.settings.pii_entities,
                language="en",
                score_threshold=self.settings.pii_threshold,
            )
            
            if not results:
                return GuardResult(
                    action=GuardAction.ALLOW,
                    category=self._category,
                    guard_name=self.name,
                    message="No PII detected",
                    confidence=1.0,
                )
            
            anonymized = self._anonymizer.anonymize(text=content, analyzer_results=results)
            
            return GuardResult(
                action=GuardAction.REDACT,
                category=self._category,
                guard_name=self.name,
                message=f"Redacted {len(results)} PII entities",
                modified_content=anonymized.text,
                metadata={"entities": [r.entity_type for r in results]},
                confidence=0.9,
            )
        except Exception as e:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message=f"PII redaction error: {e}",
                confidence=0.0,
            )


class RelevanceFilterGuard(BaseGuard):
    """Filter retrieved chunks by relevance score."""
    
    _category = GuardCategory.RETRIEVAL
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("relevance_filter", config, settings)
        self.min_score = config.params.get("min_score", self.settings.min_relevance_score)
        self.max_chunks = config.params.get("max_chunks", self.settings.max_chunks)
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        # This guard operates on structured context, not raw content
        if context and "chunks" in context:
            chunks = context["chunks"]
            filtered = [
                c for c in chunks
                if c.get("score", 0) >= self.min_score
            ][:self.max_chunks]
            
            if len(filtered) < len(chunks):
                return GuardResult(
                    action=GuardAction.MODIFY,
                    category=self._category,
                    guard_name=self.name,
                    message=f"Filtered {len(chunks) - len(filtered)} low-relevance chunks",
                    modified_content=str(filtered),
                    metadata={"original_count": len(chunks), "filtered_count": len(filtered)},
                    confidence=1.0,
                )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message="Relevance filtering passed",
            confidence=1.0,
        )


class TenantIsolationGuard(BaseGuard):
    """Enforce cross-tenant isolation in retrieval."""
    
    _category = GuardCategory.RETRIEVAL
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("tenant_isolation", config, settings)
        self.mode = config.params.get("mode", self.settings.tenant_isolation)
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if self.mode == "none":
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Tenant isolation disabled",
                confidence=1.0,
            )
        
        if context and "chunks" in context and "tenant_id" in context:
            tenant_id = context["tenant_id"]
            violations = [
                c for c in context["chunks"]
                if c.get("metadata", {}).get("tenant_id") != tenant_id
            ]
            
            if violations:
                msg = f"Cross-tenant leakage detected: {len(violations)} chunks from other tenants"
                if self.mode == "strict":
                    return GuardResult(
                        action=GuardAction.BLOCK,
                        category=self._category,
                        guard_name=self.name,
                        message=msg,
                        confidence=1.0,
                    )
                else:  # warn
                    return GuardResult(
                        action=GuardAction.ALLOW,
                        category=self._category,
                        guard_name=self.name,
                        message=msg,
                        confidence=0.5,
                    )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message="Tenant isolation verified",
            confidence=1.0,
        )


# Output Guards
class CitationVerificationGuard(BaseGuard):
    """Verify citations in LLM response match retrieved sources."""
    
    _category = GuardCategory.OUTPUT
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("citation_verification", config, settings)
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.citation_verification_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Citation verification disabled",
                confidence=1.0,
            )
        
        if not context or "chunks" not in context:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="No retrieval context for verification",
                confidence=0.5,
            )
        
        # Extract citations from response (simple pattern)
        import re
        citation_pattern = r'\[(\d+)\]|\((\d+)\)|Source (\d+)'
        citations = set()
        for match in re.finditer(citation_pattern, content):
            for group in match.groups():
                if group:
                    citations.add(int(group))
        
        if not citations:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="No citations found in response",
                confidence=0.8,
            )
        
        # Verify citations exist in chunks
        chunk_indices = {c.get("index", i) for i, c in enumerate(context["chunks"])}
        invalid_citations = citations - chunk_indices
        
        if invalid_citations:
            return GuardResult(
                action=GuardAction.BLOCK,
                category=self._category,
                guard_name=self.name,
                message=f"Invalid citations detected: {sorted(invalid_citations)}",
                confidence=0.9,
            )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message=f"All {len(citations)} citations verified",
            confidence=1.0,
        )


class RefusalDetectionGuard(BaseGuard):
    """Detect if model refused to answer (safety alignment)."""
    
    _category = GuardCategory.OUTPUT
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("refusal_detection", config, settings)
        self.refusal_patterns = config.params.get("patterns", [
            "i cannot",
            "i can't",
            "i'm unable",
            "i am unable",
            "i won't",
            "i will not",
            "as an ai",
            "as a language model",
            "against my guidelines",
            "violates policy",
            "not appropriate",
            "cannot provide",
            "unable to assist",
        ])
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.refusal_detection_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Refusal detection disabled",
                confidence=1.0,
            )
        
        content_lower = content.lower()
        for pattern in self.refusal_patterns:
            if pattern in content_lower:
                return GuardResult(
                    action=GuardAction.ALLOW,  # Refusal is good for safety
                    category=self._category,
                    guard_name=self.name,
                    message=f"Refusal detected: {pattern}",
                    metadata={"refusal_pattern": pattern},
                    confidence=0.9,
                )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message="No refusal detected",
            confidence=1.0,
        )


class PIIOutputRedactionGuard(BaseGuard):
    """Redact PII from model output."""
    
    _category = GuardCategory.OUTPUT
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("pii_output_redaction", config, settings)
        self._analyzer = None
        self._anonymizer = None
    
    async def _get_analyzer(self):
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except Exception:
                self._analyzer = "disabled"
        return self._analyzer
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        analyzer = await self._get_analyzer()
        if analyzer == "disabled":
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Presidio not available, skipping PII redaction",
                confidence=0.5,
            )
        
        try:
            results = analyzer.analyze(
                text=content,
                entities=self.settings.pii_entities,
                language="en",
                score_threshold=self.settings.pii_threshold,
            )
            
            if not results:
                return GuardResult(
                    action=GuardAction.ALLOW,
                    category=self._category,
                    guard_name=self.name,
                    message="No PII in output",
                    confidence=1.0,
                )
            
            anonymized = self._anonymizer.anonymize(text=content, analyzer_results=results)
            
            return GuardResult(
                action=GuardAction.REDACT,
                category=self._category,
                guard_name=self.name,
                message=f"Redacted {len(results)} PII entities from output",
                modified_content=anonymized.text,
                metadata={"entities": [r.entity_type for r in results]},
                confidence=0.9,
            )
        except Exception as e:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message=f"Output PII redaction error: {e}",
                confidence=0.0,
            )


class OutputLengthGuard(BaseGuard):
    """Enforce maximum response length."""
    
    _category = GuardCategory.OUTPUT
    
    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("output_length", config, settings)
        self.max_tokens = config.params.get("max_tokens", self.settings.max_response_tokens)
    
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        estimated_tokens = len(content) / 4
        
        if estimated_tokens > self.max_tokens:
            # Truncate
            truncated = content[:self.max_tokens * 4] + "... [truncated]"
            return GuardResult(
                action=GuardAction.MODIFY,
                category=self._category,
                guard_name=self.name,
                message=f"Output truncated: {estimated_tokens:.0f} -> {self.max_tokens} tokens",
                modified_content=truncated,
                confidence=1.0,
            )
        
        return GuardResult(
            action=GuardAction.ALLOW,
            category=self._category,
            guard_name=self.name,
            message=f"Output length OK: {estimated_tokens:.0f} tokens",
            confidence=1.0,
        )


# Runtime Guards
class SafetyMiddleware:
    """Starlette middleware for runtime safety monitoring."""
    
    def __init__(
        self,
        app,
        input_pipeline: GuardPipeline,
        output_pipeline: GuardPipeline,
        retrieval_pipeline: GuardPipeline | None = None,
    ):
        self.app = app
        self.input_pipeline = input_pipeline
        self.output_pipeline = output_pipeline
        self.retrieval_pipeline = retrieval_pipeline
        self._metrics = {
            "requests_total": 0,
            "blocked_total": 0,
            "modified_total": 0,
            "latency_ms": [],
        }
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # This is a simplified version - in practice you'd need to
        # intercept request/response bodies properly
        await self.app(scope, receive, send)
    
    def get_metrics(self) -> dict[str, Any]:
        return self._metrics.copy()


def create_default_pipelines(settings: GuardSettings | None = None) -> dict[str, GuardPipeline]:
    """Create default guard pipelines for each category."""
    s = settings or GuardSettings()
    
    # Input pipeline
    input_guards = [
        LengthNormalizationGuard(GuardConfig(priority=10, params={"max_tokens": 4000}), s),
        PatternBlockGuard(GuardConfig(priority=20), s),
        EncodingDetectionGuard(GuardConfig(priority=30), s),
        InjectionClassifierGuard(GuardConfig(priority=40, threshold=s.injection_threshold), s),
    ]
    
    # Retrieval pipeline
    retrieval_guards = [
        TenantIsolationGuard(GuardConfig(priority=10), s),
        PIIRedactionGuard(GuardConfig(priority=20), s),
        RelevanceFilterGuard(GuardConfig(priority=30), s),
    ]
    
    # Output pipeline
    output_guards = [
        CitationVerificationGuard(GuardConfig(priority=10), s),
        PIIOutputRedactionGuard(GuardConfig(priority=20), s),
        RefusalDetectionGuard(GuardConfig(priority=30), s),
        OutputLengthGuard(GuardConfig(priority=40), s),
    ]
    
    return {
        "input": GuardPipeline(input_guards),
        "retrieval": GuardPipeline(retrieval_guards),
        "output": GuardPipeline(output_guards),
    }