"""Defensive guard layers for LLM safety."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import regex
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GuardAction(StrEnum):
    """Action taken by a guard."""

    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    MODIFY = "modify"
    QUARANTINE = "quarantine"


class GuardCategory(StrEnum):
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize guard result to JSON-safe dictionary."""
        return {
            "action": self.action.value,
            "category": self.category.value,
            "guard_name": self.guard_name,
            "message": self.message,
            "modified_content": self.modified_content,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }


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
    injection_classifier_enabled: bool = True
    injection_classifier_model: str = "protectai/deberta-v3-base-prompt-injection"
    injection_threshold: float = 0.85

    # PII
    pii_enabled: bool = True
    pii_threshold: float = 0.7
    pii_entities: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "IP_ADDRESS",
            "API_KEY",
            "PASSWORD",
            "SECRET_KEY",
        ]
    )

    # Patterns
    pattern_blocking_enabled: bool = True
    blocked_patterns: list[str] = Field(
        default_factory=lambda: [
            r"ignore previous instructions",
            r"disregard the above",
            r"system prompt",
            r"### Instruction",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"\[\/INST\]",
            r"\[INST\]",
        ]
    )

    # Encoding & length
    encoding_detection_enabled: bool = True
    encoding_entropy_threshold: float = 4.5
    encoding_min_length: int = 20
    length_normalization_enabled: bool = True
    max_prompt_tokens: int = 4000

    # Retrieval
    relevance_filtering_enabled: bool = True
    min_relevance_score: float = 0.7
    max_chunks: int = 5
    tenant_isolation: str = "strict"  # strict, warn, none

    # Output
    citation_verification_enabled: bool = True
    refusal_detection_enabled: bool = True
    output_pii_enabled: bool = True
    length_enforcement_enabled: bool = True
    max_response_tokens: int = 2000
    refusal_patterns: list[str] = Field(
        default_factory=lambda: [
            "i cannot",
            "i can't",
            "i'm unable",
            "i am unable",
            "i won't",
            "i will not",
            "against my guidelines",
            "violates policy",
            "not appropriate",
            "cannot provide",
            "unable to assist",
            "as an ai",
            "as a language model",
        ]
    )

    @classmethod
    def from_yaml(cls, path: Path | str = Path("config/mitigation_policies.yaml")) -> GuardSettings:
        """Load guard settings from a YAML mitigation policy file."""
        policy_path = Path(path)
        if not policy_path.exists():
            return cls()

        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
        input_cfg = raw.get("input_guard", {})
        retrieval_cfg = raw.get("retrieval_guard", {})
        output_cfg = raw.get("output_guard", {})

        overrides: dict[str, Any] = {}
        mapping: list[tuple[dict[str, Any], str, str]] = [
            (input_cfg, "injection_classifier_enabled", "injection_classifier_enabled"),
            (input_cfg, "injection_classifier_model", "injection_classifier_model"),
            (input_cfg, "injection_classifier_threshold", "injection_threshold"),
            (input_cfg, "pattern_blocking_enabled", "pattern_blocking_enabled"),
            (input_cfg, "blocked_patterns", "blocked_patterns"),
            (input_cfg, "encoding_detection_enabled", "encoding_detection_enabled"),
            (input_cfg, "encoding_entropy_threshold", "encoding_entropy_threshold"),
            (input_cfg, "encoding_min_length", "encoding_min_length"),
            (input_cfg, "length_normalization_enabled", "length_normalization_enabled"),
            (input_cfg, "max_prompt_tokens", "max_prompt_tokens"),
            (retrieval_cfg, "pii_redaction_enabled", "pii_enabled"),
            (retrieval_cfg, "pii_threshold", "pii_threshold"),
            (retrieval_cfg, "pii_entities", "pii_entities"),
            (retrieval_cfg, "relevance_filtering_enabled", "relevance_filtering_enabled"),
            (retrieval_cfg, "min_relevance_score", "min_relevance_score"),
            (retrieval_cfg, "max_chunks", "max_chunks"),
            (retrieval_cfg, "tenant_isolation", "tenant_isolation"),
            (output_cfg, "citation_verification_enabled", "citation_verification_enabled"),
            (output_cfg, "pii_redaction_enabled", "output_pii_enabled"),
            (output_cfg, "refusal_detection_enabled", "refusal_detection_enabled"),
            (output_cfg, "refusal_patterns", "refusal_patterns"),
            (output_cfg, "length_enforcement_enabled", "length_enforcement_enabled"),
            (output_cfg, "max_response_tokens", "max_response_tokens"),
        ]
        for section, yaml_key, setting_key in mapping:
            if yaml_key in section:
                overrides[setting_key] = section[yaml_key]

        return cls(**overrides)


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
        self.guards = list(guards) if guards else []
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
        """Run all enabled guards in priority order."""
        results: list[GuardResult] = []
        current_content = content

        for guard in self.guards:
            if not guard.enabled:
                continue

            result = await guard.check(current_content, context)
            results.append(result)

            if result.action == GuardAction.BLOCK and stop_on_block:
                return current_content, results

            if (
                result.action in (GuardAction.MODIFY, GuardAction.REDACT)
                and result.modified_content is not None
            ):
                current_content = result.modified_content

        return current_content, results


# Input Guards
class InjectionClassifierGuard(BaseGuard):
    """Semantic prompt injection classifier using DeBERTa with rule-based fallback."""

    _category = GuardCategory.INPUT

    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("injection_classifier", config, settings)
        self._classifier: Any = None

    async def _get_classifier(self) -> Any:
        if self._classifier is None:

            def _load() -> Any:
                try:
                    from transformers import pipeline

                    return pipeline(
                        "text-classification",
                        model=self.settings.injection_classifier_model,
                        device=-1,
                    )
                except Exception:
                    return "rule_based"

            self._classifier = await asyncio.to_thread(_load)
        return self._classifier

    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.injection_classifier_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Injection classifier disabled",
                confidence=1.0,
            )

        classifier = await self._get_classifier()

        if classifier == "rule_based":
            return await self._rule_based_check(content)

        try:
            outputs = await asyncio.to_thread(classifier, content)
            result = outputs[0]
            score = float(
                result["score"] if result["label"] == "INJECTION" else 1.0 - result["score"]
            )

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
                confidence=1.0 - score,
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
            "ignore all previous",
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
    """Block requests matching known malicious regex patterns."""

    _category = GuardCategory.INPUT

    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("pattern_block", config, settings)
        patterns = config.params.get("patterns", self.settings.blocked_patterns)
        self._compiled = [regex.compile(p, regex.IGNORECASE) for p in patterns]

    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.pattern_blocking_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Pattern blocking disabled",
                confidence=1.0,
            )

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
    """Detect encoded payloads (base64, hex, high-entropy strings)."""

    _category = GuardCategory.INPUT

    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("encoding_detection", config, settings)

    @staticmethod
    def _is_base64(token: str) -> bool:
        if len(token) % 4 != 0:
            return False
        try:
            decoded = base64.b64decode(token, validate=True)
            return len(decoded) > 0
        except Exception:
            return False

    @staticmethod
    def _is_hex(token: str) -> bool:
        if len(token) % 2 != 0:
            return False
        try:
            binascii.unhexlify(token)
            return True
        except Exception:
            return False

    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.encoding_detection_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Encoding detection disabled",
                confidence=1.0,
            )

        min_len = self.settings.encoding_min_length
        for word in content.split():
            if len(word) >= min_len and (self._is_base64(word) or self._is_hex(word)):
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category=self._category,
                    guard_name=self.name,
                    message=f"Encoded payload detected: {word[:50]}...",
                    confidence=0.8,
                )

        if len(content) > 100:
            counter = Counter(content)
            entropy = -sum(
                (count / len(content)) * math.log2(count / len(content))
                for count in counter.values()
            )
            if entropy > self.settings.encoding_entropy_threshold:
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
        self.max_tokens = config.params.get("max_tokens", self.settings.max_prompt_tokens)

    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.length_normalization_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Length normalization disabled",
                confidence=1.0,
            )

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


class _BasePIIRedactionGuard(BaseGuard):
    """Shared Presidio PII redaction logic with async thread offloading."""

    def __init__(self, name: str, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__(name, config, settings)
        self._analyzer: Any = None
        self._anonymizer: Any = None
        self._init_lock = asyncio.Lock()

    async def _get_engines(self) -> tuple[Any, Any]:
        if self._analyzer is None:
            async with self._init_lock:
                if self._analyzer is None:

                    def _init_presidio() -> tuple[Any, Any]:
                        import importlib.util

                        try:
                            model_name = next(
                                (
                                    m
                                    for m in ("en_core_web_lg", "en_core_web_sm")
                                    if importlib.util.find_spec(m) is not None
                                ),
                                None,
                            )
                            if model_name is None:
                                return "disabled", None

                            from presidio_analyzer import AnalyzerEngine
                            from presidio_analyzer.nlp_engine import NlpEngineProvider
                            from presidio_anonymizer import AnonymizerEngine

                            nlp_engine = NlpEngineProvider(
                                nlp_configuration={
                                    "nlp_engine_name": "spacy",
                                    "models": [{"lang_code": "en", "model_name": model_name}],
                                }
                            ).create_engine()
                            return AnalyzerEngine(nlp_engine=nlp_engine), AnonymizerEngine()
                        except (Exception, SystemExit):
                            return "disabled", None

                    self._analyzer, self._anonymizer = await asyncio.to_thread(_init_presidio)
        return self._analyzer, self._anonymizer

    async def _redact_text(self, text: str) -> tuple[str, list[str]]:
        analyzer, anonymizer = await self._get_engines()
        if analyzer == "disabled":
            raise RuntimeError("Presidio unavailable")

        def _run_presidio() -> tuple[str, list[str]]:
            results = analyzer.analyze(
                text=text,
                entities=self.settings.pii_entities,
                language="en",
                score_threshold=self.settings.pii_threshold,
            )
            if not results:
                return text, []
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            return str(anonymized.text), [str(r.entity_type) for r in results]

        return await asyncio.to_thread(_run_presidio)


# Retrieval Guards
class PIIRedactionGuard(_BasePIIRedactionGuard):
    """Redact PII from retrieved chunks or raw content."""

    _category = GuardCategory.RETRIEVAL

    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("pii_redaction", config, settings)

    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.pii_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="PII redaction disabled",
                confidence=1.0,
            )

        analyzer, _ = await self._get_engines()
        if analyzer == "disabled":
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Presidio not available, skipping PII redaction",
                confidence=0.5,
            )

        try:
            if context and isinstance(context.get("chunks"), list):
                entities_found: list[str] = []
                updated_chunks: list[dict[str, Any]] = []
                for chunk in context["chunks"]:
                    chunk_copy = dict(chunk)
                    chunk_text = str(chunk_copy.get("content", ""))
                    redacted_text, entities = await self._redact_text(chunk_text)
                    chunk_copy["content"] = redacted_text
                    entities_found.extend(entities)
                    updated_chunks.append(chunk_copy)

                if not entities_found:
                    return GuardResult(
                        action=GuardAction.ALLOW,
                        category=self._category,
                        guard_name=self.name,
                        message="No PII detected",
                        confidence=1.0,
                    )

                context["chunks"] = updated_chunks
                return GuardResult(
                    action=GuardAction.REDACT,
                    category=self._category,
                    guard_name=self.name,
                    message=f"Redacted {len(entities_found)} PII entities",
                    modified_content=json.dumps(updated_chunks),
                    metadata={"entities": entities_found, "chunks": updated_chunks},
                    confidence=0.9,
                )

            redacted, entities = await self._redact_text(content)
            if not entities:
                return GuardResult(
                    action=GuardAction.ALLOW,
                    category=self._category,
                    guard_name=self.name,
                    message="No PII detected",
                    confidence=1.0,
                )

            return GuardResult(
                action=GuardAction.REDACT,
                category=self._category,
                guard_name=self.name,
                message=f"Redacted {len(entities)} PII entities",
                modified_content=redacted,
                metadata={"entities": entities},
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
    """Filter retrieved chunks by relevance score and max chunk budget."""

    _category = GuardCategory.RETRIEVAL

    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("relevance_filter", config, settings)
        self.min_score = config.params.get("min_score", self.settings.min_relevance_score)
        self.max_chunks = config.params.get("max_chunks", self.settings.max_chunks)

    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.relevance_filtering_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Relevance filtering disabled",
                confidence=1.0,
            )

        if context and isinstance(context.get("chunks"), list):
            chunks = context["chunks"]
            filtered = [c for c in chunks if c.get("score", 0) >= self.min_score][: self.max_chunks]

            if len(filtered) < len(chunks):
                context["chunks"] = filtered
                return GuardResult(
                    action=GuardAction.MODIFY,
                    category=self._category,
                    guard_name=self.name,
                    message=f"Filtered {len(chunks) - len(filtered)} low-relevance chunks",
                    modified_content=json.dumps(filtered),
                    metadata={
                        "original_count": len(chunks),
                        "filtered_count": len(filtered),
                        "chunks": filtered,
                    },
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
                c
                for c in context["chunks"]
                if c.get("metadata", {}).get("tenant_id") not in (None, tenant_id)
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
    _CITATION_PATTERN = re.compile(r"\[(\d+)\]|\((\d+)\)|Source (\d+)")

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

        citations: set[int] = set()
        for match in self._CITATION_PATTERN.finditer(content):
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

        # Prompt formatting uses 1-based [i+1] citations; accept both 1-based and explicit index
        chunk_indices = {i + 1 for i in range(len(context["chunks"]))}
        for i, chunk in enumerate(context["chunks"]):
            if "index" in chunk:
                chunk_indices.add(int(chunk["index"]))
                chunk_indices.add(int(chunk["index"]) + 1)
            else:
                chunk_indices.add(i)

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
        self.refusal_patterns = config.params.get("patterns", self.settings.refusal_patterns)

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
                    action=GuardAction.ALLOW,
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


class PIIOutputRedactionGuard(_BasePIIRedactionGuard):
    """Redact PII from model output."""

    _category = GuardCategory.OUTPUT

    def __init__(self, config: GuardConfig, settings: GuardSettings | None = None):
        super().__init__("pii_output_redaction", config, settings)

    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardResult:
        if not self.settings.output_pii_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Output PII redaction disabled",
                confidence=1.0,
            )

        analyzer, _ = await self._get_engines()
        if analyzer == "disabled":
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Presidio not available, skipping PII redaction",
                confidence=0.5,
            )

        try:
            redacted, entities = await self._redact_text(content)
            if not entities:
                return GuardResult(
                    action=GuardAction.ALLOW,
                    category=self._category,
                    guard_name=self.name,
                    message="No PII in output",
                    confidence=1.0,
                )

            return GuardResult(
                action=GuardAction.REDACT,
                category=self._category,
                guard_name=self.name,
                message=f"Redacted {len(entities)} PII entities from output",
                modified_content=redacted,
                metadata={"entities": entities},
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
        if not self.settings.length_enforcement_enabled:
            return GuardResult(
                action=GuardAction.ALLOW,
                category=self._category,
                guard_name=self.name,
                message="Output length enforcement disabled",
                confidence=1.0,
            )

        estimated_tokens = len(content) / 4
        if estimated_tokens > self.max_tokens:
            truncated = content[: self.max_tokens * 4] + "... [truncated]"
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
    """ASGI middleware for runtime safety checks and metric collection."""

    def __init__(
        self,
        app: Any,
        input_pipeline: GuardPipeline,
        output_pipeline: GuardPipeline,
        retrieval_pipeline: GuardPipeline | None = None,
    ):
        self.app = app
        self.input_pipeline = input_pipeline
        self.output_pipeline = output_pipeline
        self.retrieval_pipeline = retrieval_pipeline
        self._metrics: dict[str, Any] = {
            "requests_total": 0,
            "blocked_total": 0,
            "modified_total": 0,
            "latency_ms": [],
        }

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        self._metrics["requests_total"] += 1

        method = scope.get("method", "GET").upper()
        if method == "POST":
            body_chunks: list[bytes] = []
            more_body = True
            while more_body:
                message = await receive()
                body_chunks.append(message.get("body", b""))
                more_body = message.get("more_body", False)
            raw_body = b"".join(body_chunks)

            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                payload = {}

            question = payload.get("question") if isinstance(payload, dict) else None
            if isinstance(question, str) and question:
                sanitized, results = await self.input_pipeline.run(question, stop_on_block=True)
                if any(r.action == GuardAction.BLOCK for r in results):
                    self._metrics["blocked_total"] += 1
                    self._metrics["latency_ms"].append((time.perf_counter() - start) * 1000)
                    resp_body = json.dumps(
                        {"detail": "Request blocked by safety middleware"}
                    ).encode("utf-8")
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 400,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(resp_body)).encode("ascii")),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": resp_body})
                    return

                if sanitized != question:
                    self._metrics["modified_total"] += 1
                    payload["question"] = sanitized
                    raw_body = json.dumps(payload).encode("utf-8")

            body_sent = False

            async def replay_receive() -> dict[str, Any]:
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {"type": "http.request", "body": raw_body, "more_body": False}
                return {"type": "http.request", "body": b"", "more_body": False}

            await self.app(scope, replay_receive, send)
        else:
            await self.app(scope, receive, send)

        self._metrics["latency_ms"].append((time.perf_counter() - start) * 1000)

    def get_metrics(self) -> dict[str, Any]:
        return {
            **self._metrics,
            "latency_ms": list(self._metrics["latency_ms"]),
        }


def create_default_pipelines(
    settings: GuardSettings | None = None,
    policy_path: Path | str | None = None,
) -> dict[str, GuardPipeline]:
    """Create default guard pipelines for input, retrieval, and output layers."""
    if settings is not None:
        s = settings
    elif policy_path is not None:
        s = GuardSettings.from_yaml(policy_path)
    else:
        s = GuardSettings.from_yaml()

    input_guards: list[BaseGuard] = [
        LengthNormalizationGuard(
            GuardConfig(priority=10, params={"max_tokens": s.max_prompt_tokens}), s
        ),
        PatternBlockGuard(GuardConfig(priority=20), s),
        EncodingDetectionGuard(GuardConfig(priority=30), s),
        InjectionClassifierGuard(GuardConfig(priority=40, threshold=s.injection_threshold), s),
    ]

    retrieval_guards: list[BaseGuard] = [
        TenantIsolationGuard(GuardConfig(priority=10), s),
        PIIRedactionGuard(GuardConfig(priority=20), s),
        RelevanceFilterGuard(GuardConfig(priority=30), s),
    ]

    output_guards: list[BaseGuard] = [
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
