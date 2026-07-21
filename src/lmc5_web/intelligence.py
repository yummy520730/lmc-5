from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import httpx


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int

    def embed(self, text: str, *, task: str = "document") -> list[float]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


@dataclass(slots=True)
class LocalHashEmbedding:
    """Small private fallback embedding for a 2C2G deployment.

    This is deliberately described as a lexical vector, not a neural semantic
    embedding. It keeps Nap and vector plumbing useful without sending private
    memories off the server. A Gemini provider can replace it later without a
    schema migration.
    """

    dimension: int = 384
    name: str = "local_hash"
    model: str = "char-ngram-v1"

    @staticmethod
    def _features(text: str) -> list[str]:
        clean = re.sub(r"\s+", " ", (text or "").casefold()).strip()
        features: list[str] = []
        cjk_runs = re.findall(r"[\u3400-\u9fff]+", clean)
        for run in cjk_runs:
            features.extend(run)
            for size in (2, 3, 4):
                features.extend(run[index : index + size] for index in range(len(run) - size + 1))
        features.extend(re.findall(r"[a-z0-9_\-]{2,}", clean))
        return features or [clean or "empty"]

    def embed(self, text: str, *, task: str = "document") -> list[float]:
        vector = [0.0] * self.dimension
        for feature in self._features(text):
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=16
            ).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * (1.0 + min(3.0, len(feature) / 4.0))
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 8) for value in vector]


@dataclass(slots=True)
class GeminiEmbedding:
    api_key: str
    model: str = "gemini-embedding-001"
    dimension: int = 768
    name: str = "gemini"
    timeout_seconds: float = 45.0

    def embed(self, text: str, *, task: str = "document") -> list[float]:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent"
        )
        payload: dict[str, Any] = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text[:30000]}]},
            "outputDimensionality": self.dimension,
        }
        if task == "document":
            payload["taskType"] = "RETRIEVAL_DOCUMENT"
        elif task == "query":
            payload["taskType"] = "RETRIEVAL_QUERY"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                endpoint,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        values = response.json().get("embedding", {}).get("values")
        if not isinstance(values, list) or not values:
            raise RuntimeError("Gemini embedding response did not contain values")
        vector = [float(value) for value in values]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass(frozen=True, slots=True)
class DreamCandidate:
    candidate_key: str
    title: str
    content: str
    category: str
    thread: str
    importance: float
    privacy_scope: str
    protected: bool
    evidence_event_ids: tuple[int, ...]
    relation_terms: tuple[str, ...] = ()
    proposer: str = "local_evidence"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "thread": self.thread,
            "importance": self.importance,
            "privacy_scope": self.privacy_scope,
            "protected": self.protected,
            "evidence_event_ids": list(self.evidence_event_ids),
            "relation_terms": list(self.relation_terms),
            "proposer": self.proposer,
        }


_SENSITIVE_TERMS = (
    "经期", "月经", "怀孕", "流血", "疼痛", "药", "医院", "诊断", "创伤",
    "性侵", "家暴", "离婚", "赔偿", "律师", "法院", "隐私", "密码", "token",
)
_RELATIONSHIP_TERMS = (
    "关系", "感情", "爱", "亲密", "依恋", "吃醋", "承诺", "纪念", "老公", "老婆",
)
_IDENTITY_TERMS = ("身份", "名字", "称呼", "规则", "不要叫", "必须", "我是", "她叫")
_TASK_TERMS = ("待办", "计划", "下一步", "要做", "部署", "修复", "项目")

_CREDENTIAL_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\b(?:sk|AIza|ghp|github_pat)[-_A-Za-z0-9]{12,}\b", re.IGNORECASE),
    re.compile(
        r"\b(api[_-]?key|access[_-]?token|password|passwd|secret)\b\s*[:=]\s*"
        r"([^\s,;]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:postgres(?:ql)?|mysql|redis)://[^\s]+", re.IGNORECASE),
)


def redact_credentials(text: str) -> str:
    redacted = text
    for index, pattern in enumerate(_CREDENTIAL_PATTERNS):
        if index == 2:
            redacted = pattern.sub(r"\1=[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def _candidate_category(text: str) -> tuple[str, str, float, bool]:
    importance = 5.0
    if _contains_any(text, _IDENTITY_TERMS):
        return "identity", "identity", 8.5, True
    if _contains_any(text, _RELATIONSHIP_TERMS):
        return "relationship_moment", "relationship", 8.0, False
    if _contains_any(text, _TASK_TERMS):
        return "tasks", "projects", 6.5, False
    if _contains_any(text, _SENSITIVE_TERMS):
        return "episode", "timeline", 7.0, False
    if len(text) >= 260:
        importance += 1.0
    return "episode", "timeline", importance, False


def local_dream_candidates(events: Sequence[dict[str, Any]]) -> list[DreamCandidate]:
    """Create evidence-preserving candidates without an external model.

    No new facts are synthesized. Candidate content is a bounded list of exact
    source-event snippets so the dry-run report is auditable.
    """
    if not events:
        return []
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_session: str | None = None
    for event in events:
        session = str(event.get("session_id") or "")
        if current and (session != current_session or len(current) >= 10):
            chunks.append(current)
            current = []
        current.append(dict(event))
        current_session = session
    if current:
        chunks.append(current)

    candidates: list[DreamCandidate] = []
    for chunk in chunks:
        evidence: list[tuple[int, str]] = []
        for item in chunk:
            content = redact_credentials(str(item.get("content") or ""))
            content = re.sub(r"\s+", " ", content).strip()
            if content and item.get("id") is not None:
                evidence.append((int(item["id"]), content[:500]))
        evidence_ids = tuple(event_id for event_id, _ in evidence)
        lines = [line for _, line in evidence]
        joined = "\n".join(lines).strip()
        if not joined or not evidence_ids:
            continue
        category, thread, importance, protected = _candidate_category(joined)
        if len(joined) < 24 and importance < 7.5:
            continue
        privacy_scope = "sensitive" if _contains_any(joined, _SENSITIVE_TERMS) else "personal"
        relation_terms = tuple(
            term
            for term in (*_RELATIONSHIP_TERMS, *_TASK_TERMS, *_SENSITIVE_TERMS)
            if term.casefold() in joined.casefold()
        )[:12]
        first = lines[0].strip(" ·:：-—，,。；;") if lines else "本次会话"
        title = f"夜梦候选：{first[:54]}"
        evidence_text = "\n".join(
            f"- [event:{event_id}] {line}" for event_id, line in evidence
        )
        key_material = f"{','.join(map(str, evidence_ids))}\0{joined}"
        candidate_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:32]
        candidates.append(
            DreamCandidate(
                candidate_key=candidate_key,
                title=title,
                content=evidence_text,
                category=category,
                thread=thread,
                importance=importance,
                privacy_scope=privacy_scope,
                protected=protected and privacy_scope != "sensitive",
                evidence_event_ids=evidence_ids,
                relation_terms=relation_terms,
            )
        )
    return candidates


@dataclass(slots=True)
class GeminiDreamProposer:
    api_key: str
    model: str = "gemini-2.5-flash"
    timeout_seconds: float = 90.0

    def propose(self, events: Sequence[dict[str, Any]]) -> list[DreamCandidate]:
        if not events:
            return []
        allowed_ids = {int(item["id"]) for item in events}
        source = [
            {
                "id": int(item["id"]),
                "role": str(item.get("role") or "user"),
                "content": redact_credentials(str(item.get("content") or ""))[:1500],
            }
            for item in events
        ]
        prompt = (
            "你是只做证据整理的记忆海马体，不是创作者。只能从输入事件中提取以后确实有用的候选；"
            "不得补写、猜测或改变语气。每条候选必须列出真实 evidence_event_ids。日常寒暄不要入选。\n"
            + json.dumps(source, ensure_ascii=False)
        )
        schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": [
                    "title", "content", "category", "thread", "importance",
                    "privacy_scope", "protected", "evidence_event_ids", "relation_terms",
                ],
                "properties": {
                    "title": {"type": "STRING"},
                    "content": {"type": "STRING"},
                    "category": {"type": "STRING"},
                    "thread": {"type": "STRING"},
                    "importance": {"type": "NUMBER"},
                    "privacy_scope": {"type": "STRING"},
                    "protected": {"type": "BOOLEAN"},
                    "evidence_event_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "relation_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
            },
        }
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": 0.1,
            },
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                endpoint,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        raw_text = "".join(str(part.get("text") or "") for part in parts)
        raw_candidates = json.loads(raw_text or "[]")
        if not isinstance(raw_candidates, list):
            raise RuntimeError("Gemini dream response was not an array")

        candidates: list[DreamCandidate] = []
        for raw in raw_candidates[:20]:
            if not isinstance(raw, dict):
                continue
            evidence_ids = tuple(
                dict.fromkeys(
                    int(value)
                    for value in raw.get("evidence_event_ids") or []
                    if int(value) in allowed_ids
                )
            )
            if not evidence_ids:
                continue
            title = str(raw.get("title") or "").strip()[:180]
            content = str(raw.get("content") or "").strip()[:5000]
            if not title or not content:
                continue
            key_material = f"gemini\0{','.join(map(str, evidence_ids))}\0{title}\0{content}"
            candidates.append(
                DreamCandidate(
                    candidate_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:32],
                    title=title,
                    content=content,
                    category=str(raw.get("category") or "episode")[:80],
                    thread=str(raw.get("thread") or "timeline")[:80],
                    importance=max(1.0, min(10.0, float(raw.get("importance") or 5.0))),
                    privacy_scope=(
                        str(raw.get("privacy_scope"))
                        if str(raw.get("privacy_scope")) in {"personal", "sensitive", "public"}
                        else "personal"
                    ),
                    protected=bool(raw.get("protected", False)),
                    evidence_event_ids=evidence_ids,
                    relation_terms=tuple(str(value)[:80] for value in raw.get("relation_terms") or [])[:12],
                    proposer="gemini",
                )
            )
        return candidates
