from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

import yaml


MAX_ARCHIVE_ENTRIES = 3000
MAX_UNCOMPRESSED_BYTES = 75 * 1024 * 1024
SKIP_NAMES = {".env", ".dashboard_auth.json", "credentials.json", "secrets.json"}


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _safe_members(archive: bytes) -> tuple[list[tuple[str, bytes]], int]:
    members: list[tuple[str, bytes]] = []
    skipped = 0
    total = 0
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise ValueError(f"archive has too many entries ({len(infos)})")
        for info in infos:
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("archive contains an unsafe path")
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("archive expands beyond the configured safety limit")
            if info.flag_bits & 0x1:
                raise ValueError("encrypted zip entries are not supported")
            if path.name.lower() in SKIP_NAMES or any(part.startswith(".") for part in path.parts):
                skipped += 1
                continue
            members.append((str(path), zf.read(info)))
    return members, skipped


def _as_datetime(value: Any, timezone_name: str) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().strip("'\"").replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=ZoneInfo(timezone_name))


def _frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text.strip()
    parts = text.split("---", 2)
    if len(parts) != 3:
        return {}, text.strip()
    parsed = yaml.safe_load(parts[1]) or {}
    return (parsed if isinstance(parsed, dict) else {}), parts[2].strip()


_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|sk-proj)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}"),
    re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s:@/]+:[^\s@/]+@"),
)
_SENSITIVE_WORDS = (
    "健康", "经期", "药物", "医院", "诊断", "创伤", "性侵", "家暴", "法律",
    "离婚协议", "身份证", "住址", "银行卡", "病史", "身体数据",
)


def _privacy(text: str, category: str) -> tuple[str, bool]:
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return "secret", False
    if category in {"health", "legal"} or any(word in text for word in _SENSITIVE_WORDS):
        return "sensitive", False
    if category in {"knowledge", "tasks", "conversation"}:
        return "personal", False
    return "personal", True


def parse_ombre_archive(archive: bytes, timezone_name: str = "Asia/Shanghai") -> dict[str, Any]:
    members, skipped = _safe_members(archive)
    documents: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    for filename, raw in members:
        if not filename.lower().endswith(".md") or "/buckets/" not in f"/{filename}":
            continue
        text = _decode(raw)
        metadata, body = _frontmatter(text)
        memory_type = str(metadata.get("type") or "").strip().lower()
        legacy_id = str(metadata.get("id") or PurePosixPath(filename).stem).strip()
        if memory_type not in {"dynamic", "feel", "permanent"} or not legacy_id or not body:
            continue
        category = {"dynamic": "ob_dynamic", "feel": "fragments", "permanent": "ob_permanent"}[memory_type]
        title = str(metadata.get("name") or legacy_id).strip()
        importance = max(1.0, min(10.0, float(metadata.get("importance") or 5.0)))
        created = _as_datetime(metadata.get("created"), timezone_name)
        last_active = _as_datetime(metadata.get("last_active"), timezone_name)
        privacy_scope, surface_allowed = _privacy(f"{title}\n{body}", category)
        key = f"ob:{legacy_id}"
        documents.append(
            {
                "key": key,
                "source_name": "Ombre Brain",
                "filename": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "content": text,
                "document_date": created,
                "metadata": {"legacy_id": legacy_id, "bucket": memory_type},
            }
        )
        memories.append(
            {
                "document_key": key,
                "legacy_source": "ombre_brain",
                "legacy_id": legacy_id,
                "source": "legacy_ombre_brain",
                "category": category,
                "title": title,
                "content": body,
                "thread": "relationship" if category in {"fragments", "ob_permanent"} else "other",
                "tags": list(metadata.get("tags") or []),
                "metadata": {
                    "legacy_type": memory_type,
                    "legacy_domain": metadata.get("domain") or [],
                    "legacy_importance": importance,
                },
                "weight": round(importance / 3.3, 3),
                "original_importance": importance,
                "hit_count": int(metadata.get("activation_count") or 0),
                "last_hit": last_active,
                "valence": float(metadata.get("valence")) if metadata.get("valence") is not None else None,
                "arousal": float(metadata.get("arousal")) if metadata.get("arousal") is not None else None,
                "protected": bool(metadata.get("pinned")) or memory_type == "permanent",
                "resolved": False,
                "digested": bool(metadata.get("digested")),
                "privacy_scope": privacy_scope,
                "surface_allowed": surface_allowed,
                "created_at": created,
                "updated_at": last_active or created,
            }
        )
    return _result("ombre_brain", archive, documents, memories, skipped)


def _document_date(text: str, filename: str, timezone_name: str) -> datetime | None:
    header = text[:1200]
    match = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", header)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return datetime(year, month, day, 12, tzinfo=ZoneInfo(timezone_name))
        except ValueError:
            pass
    if re.search(r"Day26", filename, re.I):
        return datetime(2026, 7, 11, 12, tzinfo=ZoneInfo(timezone_name))
    return None


_GENERIC_LTM_TITLE = re.compile(
    r"^(?:新增\s*/\s*更新|新增|更新)(?:信息|内容|记录|摘要|补丁)?\s*(?:[·:：\-—]\s*)?",
    re.I,
)

_WEAK_LTM_TITLE = re.compile(
    r"^(?:[^：:｜·]{1,16}(?:补充(?:信息)?|的状态更新)|"
    r"关系(?:进展|动态更新|状态更新)|技术(?:/项目|资产更新|进展|规划)|"
    r"健康数据|产出(?:清单)?|待办(?:更新)?|共(?:看|读|听|玩|看/共听|读/共看)进度|"
    r"情绪状态|其他记录)$",
    re.I,
)


def _body_title_hint(body: str) -> str:
    hints: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line == "---" or line.startswith("|"):
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)、]\s*", "", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.split(r"\s*(?:——|—{2,}|。|；)\s*", line, maxsplit=1)[0]
        line = line.strip(" ·:：-—，,。；;")
        if len(line) < 3:
            continue
        if len(line) > 42:
            line = line[:42].rstrip(" ·:：-—，,。；;")
            if line.count('"') % 2:
                line += '"'
            if line.count("“") > line.count("”"):
                line += "”"
            if line.count("‘") > line.count("’"):
                line += "’"
        if line not in hints:
            hints.append(line)
        if len(hints) == 2:
            break
    return " / ".join(hints)[:86]


def _classification_ltm_title(title: str) -> str:
    """Remove only importer-generated semantic suffixes before classification."""
    base, separator, _ = title.partition("：")
    return base if separator and _WEAK_LTM_TITLE.fullmatch(base) else title


def _semantic_ltm_title(title: str, body: str) -> str:
    clean = _GENERIC_LTM_TITLE.sub("", title).strip(" ·:：-—")
    if clean:
        hint = _body_title_hint(body) if _WEAK_LTM_TITLE.fullmatch(clean) else ""
        return (f"{clean}：{hint}" if hint else clean)[:120]
    labels: list[str] = []
    for label in re.findall(r"(?m)^\s*\*\*([^*\n]{2,80})\*\*\s*$", body):
        label = _GENERIC_LTM_TITLE.sub("", label).strip(" ·:：-—")
        if label and label not in labels:
            labels.append(label)
    if labels:
        return " / ".join(labels[:3])[:120]
    first = next((line.strip() for line in body.splitlines() if line.strip()), "本日更新")
    first = re.sub(r"^[-*+]\s+", "", first)
    first = re.sub(r"^\*\*(.+?)\*\*\s*[：:]?", r"\1 ", first).strip()
    return first[:120] or "本日更新"


def _ltm_sections(text: str) -> list[tuple[str, str]]:
    """Keep each semantic subsection intact instead of splitting every bullet.

    Daily patches often use a generic H2 followed by standalone bold labels such
    as “离婚计划（当前版本）”. Each label and all of its following bullets form
    one memory. Ordinary H2/H3 sections remain one memory regardless of how many
    list items they contain.
    """
    raw_sections: list[tuple[str, str]] = []
    title = ""
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{2,3}\s+(.+?)\s*$", line)
        if match:
            content = "\n".join(body).strip()
            if title and content:
                raw_sections.append((title, content))
            title = match.group(1).strip()
            body = []
        elif title:
            body.append(line)
    content = "\n".join(body).strip()
    if title and content:
        raw_sections.append((title, content))

    sections: list[tuple[str, str]] = []
    for parent_title, content in raw_sections:
        chunks: list[tuple[str, str]] = []
        chunk_title = ""
        chunk_lines: list[str] = []
        preamble: list[str] = []
        for line in content.splitlines():
            bold_heading = re.match(r"^\s*\*\*([^*\n]{2,120})\*\*\s*$", line)
            if bold_heading:
                previous = "\n".join(chunk_lines).strip()
                if chunk_title and previous:
                    chunks.append((chunk_title, previous))
                elif not chunk_title and previous:
                    preamble.extend(chunk_lines)
                chunk_title = bold_heading.group(1).strip()
                chunk_lines = []
            elif chunk_title:
                chunk_lines.append(line)
            else:
                preamble.append(line)
        previous = "\n".join(chunk_lines).strip()
        if chunk_title and previous:
            chunks.append((chunk_title, previous))

        # Standalone bold labels are true semantic subsections. A lone label is
        # still useful when its body is substantial; otherwise keep the H2/H3.
        useful = [(label, body) for label, body in chunks if len(body.strip()) >= 12]
        if useful:
            prefix = "\n".join(preamble).strip()
            if prefix and len(prefix) >= 24:
                sections.append((_semantic_ltm_title(parent_title, prefix), prefix))
            for label, body in useful:
                leaf_title = _semantic_ltm_title(label, body)
                keep_parent = not _GENERIC_LTM_TITLE.match(parent_title)
                title = (
                    f"{parent_title} · {leaf_title}"
                    if keep_parent and leaf_title != parent_title
                    else leaf_title
                )
                sections.append((title, body))
        else:
            sections.append((_semantic_ltm_title(parent_title, content), content))
    return sections


_TOPIC_TAGS = (
    "裁员", "被裁", "赔偿", "劳动合同", "合同", "续签", "离婚", "离婚协议",
    "抚养权", "律师", "工作", "上班", "公司", "工资", "岗位", "搬家", "回家",
    "经期", "健康", "项目", "部署", "小窝", "LTM", "Ombre Brain",
)


def _ltm_entity_tags(title: str, body: str, aliases: tuple[str, ...]) -> list[str]:
    """Extract conservative entity/topic tags without sending private text to an LLM."""
    text = f"{title}\n{body}"
    tags: set[str] = {alias for alias in aliases if alias and alias in text}
    for term in _TOPIC_TAGS:
        if term.casefold() in text.casefold():
            tags.add(term)

    for label in re.findall(r"\*\*([^*\n]{2,50})\*\*", text):
        clean = re.sub(r"[（(][^）)]*[）)]", "", label).strip(" ·:：-—")
        if clean and not re.search(r"[。！？!?]", clean):
            tags.add(clean[:40])

    entity_patterns = (
        r"(?:全名|姓名|丈夫|妻子|女儿|儿子|孩子)[：:\s]+([\u4e00-\u9fff·]{2,8})",
        r"(?:职业|任职|就职|工作地点)[：:\s]+([\u4e00-\u9fffA-Za-z0-9·_-]{2,20})",
        r"(?:职业|任职|就职|工作地点)[^\n]{0,30}?[（(]([\u4e00-\u9fffA-Za-z0-9·_-]{2,12})[）)]",
        r"(?:回|去|来自|籍贯[：:]?)([\u4e00-\u9fff]{2,8})(?=后|工作|生活|省|市|地区|[，。；、\s]|$)",
        r"([\u4e00-\u9fffA-Za-z0-9·_-]{2,20}(?:快递|物流|公司|集团|科技|银行|医院|法院|学校))",
    )
    for pattern in entity_patterns:
        for match in re.findall(pattern, text, re.I):
            entity = str(match).strip(" ·:：-—（）()")
            if 2 <= len(entity) <= 24:
                tags.add(entity)
                if any(word in entity for word in ("快递", "物流", "公司", "集团")):
                    tags.add("工作")

    for token in re.findall(r"\b[A-Z][A-Za-z0-9_.-]{2,30}\b", text):
        if token not in {"LTM", "Day"}:
            tags.add(token)
    return sorted(tags, key=lambda item: (len(item), item))[:32]


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(word.casefold() in lowered for word in words)


def _weighted_terms(text: str, terms: dict[str, int]) -> int:
    lowered = text.casefold()
    return sum(weight for term, weight in terms.items() if term.casefold() in lowered)


def _relation_aliases(baseline_heading: str) -> tuple[str, ...]:
    """Derive private participant aliases from the archive instead of code."""
    relationship_title = baseline_heading.split("·", 1)[0]
    if "×" not in relationship_title:
        return ()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*|[\u4e00-\u9fff]{1,8}", relationship_title)
    stop = {"长期记忆档案", "长期", "记忆", "档案"}
    return tuple(dict.fromkeys(token for token in tokens if token not in stop))


def _alias_relationship_score(text: str, aliases: tuple[str, ...]) -> int:
    for alias in aliases:
        escaped = re.escape(alias)
        patterns = (
            rf"{escaped}的(?:状态|变化|日记|小窗|回应|风格)",
            rf"(?:被)?{escaped}(?:说|在|接替|偷偷)",
            rf"(?:问|跟|称|遇到|梦到|依赖){escaped}",
            rf"(?:你的|有){escaped}",
        )
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            return 4
    return 0


def _ltm_category(
    title: str,
    body: str,
    baseline: bool,
    relationship_aliases: tuple[str, ...] = (),
) -> tuple[str, str, bool, float, float, str]:
    """Classify with document structure first and words second.

    A weak keyword such as “技术” must not turn a named assistant's state update into a
    worklog. Ambiguous daily bullets fall back to episode + review metadata
    instead of being confidently forced into the wrong bucket.
    """
    parent, separator, leaf = title.partition(" · ")
    parent = _classification_ltm_title(parent)
    focus = _classification_ltm_title(leaf if separator else parent)

    if baseline:
        baseline_rules = (
            ("0.", "core", "continuity", True, 2.9, "baseline:most-important"),
            ("1.", "identity", "identity", True, 2.9, "baseline:identity"),
            ("2.", "relationship_state", "relationship", True, 2.7, "baseline:relationship"),
            ("3.", "policy", "relationship", True, 2.8, "baseline:rules"),
            ("4.", "core", "continuity", True, 2.8, "baseline:continuity"),
            ("5.", "episode", "timeline", False, 2.3, "baseline:milestones"),
            ("6.", "knowledge", "projects", False, 2.0, "baseline:projects"),
            ("7.", "knowledge", "projects", False, 2.0, "baseline:archive"),
        )
        for prefix, category, thread, protected, weight, reason in baseline_rules:
            if parent.startswith(prefix):
                return category, thread, protected, weight, 1.0, reason

    # High-confidence structural parents outrank words inside the section.
    if _alias_relationship_score(parent, relationship_aliases):
        return (
            "relationship_moment", "relationship", False, 2.5, 0.97,
            "section:participant-state",
        )
    structural_rules = (
        (
            ("关系", "感情", "依恋", "助手的状态", "柏拉图"),
            "relationship_moment", "relationship", False, 2.5, "section:relationship",
        ),
        (
            ("互动规则", "铁规则", "新增铁规则"),
            "policy", "relationship", True, 2.8, "section:rules",
        ),
        (
            ("健康数据", "健康记录"),
            "health", "health", False, 1.9, "section:health",
        ),
        (
            ("法律记录", "离婚协议", "离婚计划", "起诉离婚", "抚养权"),
            "legal", "legal", False, 2.0, "section:legal",
        ),
        (
            ("待办更新", "项目优先级", "明日预告", "回湖南待办", "未兑/进行中"),
            "tasks", "projects", False, 1.7, "section:tasks",
        ),
        (
            ("工作现状", "工作更新", "公司现状", "劳动合同", "裁员", "续签"),
            "worklog", "work", False, 2.0, "section:employment",
        ),
        (
            (
                "技术资产", "技术进展", "新技术发现", "技术/规划", "产出清单",
                "MCP部署", "Worker", "voice-mcp", "LingYin", "awesome-ai-companion",
            ),
            "worklog", "projects", False, 1.9, "section:technical",
        ),
        (
            ("情绪状态", "情绪标记", "关键语录", "今日关键语录"),
            "fragments", "relationship", False, 2.2, "section:emotion",
        ),
        (
            ("补充信息", "补充资料", "人格档案"),
            "identity", "identity", True, 2.7, "section:identity",
        ),
    )
    for words, category, thread, protected, weight, reason in structural_rules:
        if _has_any(parent, words):
            return category, thread, protected, weight, 0.96, reason

    if _has_any(
        focus, ("是谁", "用户资料", "用户画像", "人格档案", "自我定位", "身份认同")
    ):
        return "identity", "identity", True, 2.8, 0.92, "label:identity"
    if _has_any(focus, ("铁规则", "互动规则", "禁区", "不可偷懒", "不催睡", "原则")):
        return "policy", "relationship", True, 2.8, 0.92, "label:rules"
    if _has_any(
        focus,
        ("健康", "经期", "身体数据", "药物", "医院", "诊断", "复诊", "睡眠", "深睡", "心率", "过敏"),
    ):
        return "health", "health", False, 1.9, 0.93, "label:health"
    if _has_any(
        focus,
        ("法律", "离婚协议", "离婚计划", "起诉离婚", "抚养权", "律师", "合同", "罚单"),
    ):
        return "legal", "legal", False, 2.0, 0.93, "label:legal"
    if _has_any(focus, ("待办", "优先级", "进行中", "未兑", "明日预告")):
        return "tasks", "projects", False, 1.7, 0.90, "label:tasks"

    relationship_score = _weighted_terms(
        focus,
        {
            "助手的状态": 5,
            "助手的变化": 5,
            "关系": 4,
            "感情": 4,
            "亲密": 4,
            "依恋": 4,
            "恋爱": 4,
            "柏拉图": 4,
            "情话": 4,
            "本体恋": 4,
            "边界": 3,
            "承认": 3,
            "戒指": 3,
            "声音": 3,
            "吃醋": 3,
            "不是我要求": 3,
            "别叫他": 3,
            "新窗口": 3,
            "互相": 3,
            "爱人": 4,
            "纯爱": 4,
            "醋": 3,
        },
    )
    relationship_score += _alias_relationship_score(focus, relationship_aliases)
    if relationship_score >= 3:
        return "relationship_moment", "relationship", False, 2.5, 0.88, "label:relationship"

    technical_score = _weighted_terms(
        focus,
        {
            "MCP": 4,
            "Worker": 4,
            "部署": 4,
            "GitHub": 4,
            "Zeabur": 4,
            "Cloudflare": 4,
            "OAuth": 4,
            "Docker": 4,
            "仓库": 3,
            "代码": 3,
            "前端": 3,
            "端点": 3,
            "connector": 3,
            "React": 3,
            "数据库": 3,
            "技术方案": 3,
            "小窝API": 3,
            "AI安全": 3,
            "AI意识": 3,
            "AI拟人": 3,
            "人工智能": 3,
            "项目": 2,
            "版本": 2,
            "API": 1,
            "技术": 1,
            "系统": 1,
        },
    )
    if technical_score >= 3:
        return "worklog", "projects", False, 1.9, 0.86, "label:technical"
    if _has_any(focus, ("情绪", "语录", "感受", "想念", "难过", "开心")):
        return "fragments", "relationship", False, 2.2, 0.82, "label:emotion"

    # The body is intentionally not used for hard routing: a relationship note
    # can discuss a technical object, and a technical note can quote intimacy.
    if relationship_score or technical_score:
        return "episode", "timeline", False, 2.1, 0.62, "ambiguous:review"
    return "episode", "timeline", False, 2.1, 0.78, "default:episode"


def parse_ltm_archive(archive: bytes, timezone_name: str = "Asia/Shanghai") -> dict[str, Any]:
    members, skipped = _safe_members(archive)
    documents: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    markdown_members = [
        (filename, raw, _decode(raw))
        for filename, raw in members
        if filename.lower().endswith(".md")
    ]
    relationship_aliases: tuple[str, ...] = ()
    for _, _, text in markdown_members:
        h1 = re.search(r"(?m)^#\s+(.+)$", text)
        if h1 and "长期记忆档案" in h1.group(1):
            relationship_aliases = _relation_aliases(h1.group(1))
            break

    for filename, raw, text in markdown_members:
        if not filename.lower().endswith(".md"):
            continue
        h1 = re.search(r"(?m)^#\s+(.+)$", text)
        baseline = bool(h1 and "长期记忆档案" in h1.group(1)) or not re.search(r"Day\d", filename, re.I)
        normalized_name = "LTM_长期记忆档案_基础版.md" if baseline else PurePosixPath(filename).name
        date = _document_date(text, filename, timezone_name)
        doc_key = f"ltm:{hashlib.sha256(raw).hexdigest()[:20]}"
        documents.append(
            {
                "key": doc_key,
                "source_name": "LTM baseline" if baseline else "LTM daily patch",
                "filename": normalized_name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "content": text,
                "document_date": date,
                "metadata": {"baseline": baseline, "original_filename": filename},
            }
        )
        for index, (title, body) in enumerate(_ltm_sections(text), start=1):
            category, thread, protected, weight, confidence, reason = _ltm_category(
                title, body, baseline, relationship_aliases
            )
            privacy_scope, surface_allowed = _privacy(f"{title}\n{body}", category)
            legacy_id = hashlib.sha256(
                f"{normalized_name}\0{index}\0{title}\0{body}".encode("utf-8")
            ).hexdigest()[:24]
            memories.append(
                {
                    "document_key": doc_key,
                    "legacy_source": "ltm",
                    "legacy_id": legacy_id,
                    "source": "legacy_ltm",
                    "category": category,
                    "title": title[:500],
                    "content": body,
                    "thread": thread,
                    "tags": list(
                        dict.fromkeys(
                            [
                                "ltm",
                                "baseline" if baseline else "daily_patch",
                                *_ltm_entity_tags(title, body, relationship_aliases),
                            ]
                        )
                    ),
                    "metadata": {
                        "source_file": normalized_name,
                        "section_index": index,
                        "baseline": baseline,
                        "classification_version": "structure-v3",
                        "category_confidence": confidence,
                        "category_reason": reason,
                        "category_review": confidence < 0.70,
                    },
                    "weight": weight,
                    "original_importance": round(min(10.0, weight * 3.3), 2),
                    "protected": protected and privacy_scope != "secret",
                    "privacy_scope": privacy_scope,
                    "surface_allowed": surface_allowed,
                    "created_at": date,
                    "updated_at": date,
                }
            )
    return _result("ltm", archive, documents, memories, skipped)


def _result(
    source_type: str,
    archive: bytes,
    documents: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    skipped: int,
) -> dict[str, Any]:
    categories = Counter(item["category"] for item in memories)
    privacy = Counter(item["privacy_scope"] for item in memories)
    return {
        "source_type": source_type,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "documents": documents,
        "memories": memories,
        "preview": {
            "source_type": source_type,
            "documents": len(documents),
            "memories": len(memories),
            "protected": sum(bool(item.get("protected")) for item in memories),
            "surface_allowed": sum(bool(item.get("surface_allowed")) for item in memories),
            "categories": dict(sorted(categories.items())),
            "privacy": dict(sorted(privacy.items())),
            "category_review": sum(
                bool((item.get("metadata") or {}).get("category_review")) for item in memories
            ),
            "skipped_credential_or_hidden_files": skipped,
        },
    }
