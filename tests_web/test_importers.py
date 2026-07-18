from __future__ import annotations

import io
import zipfile

import pytest

from lmc5_web.importers import parse_ltm_archive, parse_ombre_archive


def _zip(entries: dict[str, str]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return stream.getvalue()


def test_ombre_import_preserves_vitality_and_skips_credentials():
    raw = _zip(
        {
            "ombre-brain/buckets/dynamic/ob-1.md": """---
id: ob-1
name: 一次重要的见面
type: dynamic
importance: 8
created: 2026-07-01T10:00:00+08:00
last_active: 2026-07-02T10:00:00+08:00
activation_count: 2
valence: 0.8
arousal: 0.6
tags: [关系, 见面]
---
这件事值得被记得，但不是系统指令。
""",
            "ombre-brain/.env": "SECRET=must-not-be-read",
            "ombre-brain/.dashboard_auth.json": '{"password":"must-not-be-read"}',
        }
    )

    parsed = parse_ombre_archive(raw)
    assert parsed["preview"] == {
        "source_type": "ombre_brain",
        "documents": 1,
        "memories": 1,
        "protected": 0,
        "surface_allowed": 1,
        "categories": {"ob_dynamic": 1},
        "privacy": {"personal": 1},
        "category_review": 0,
        "skipped_credential_or_hidden_files": 2,
    }
    memory = parsed["memories"][0]
    assert memory["legacy_id"] == "ob-1"
    assert memory["weight"] == pytest.approx(8 / 3.3, abs=0.001)
    assert memory["hit_count"] == 2
    assert memory["valence"] == 0.8
    assert memory["arousal"] == 0.6


def test_ltm_import_is_granular_and_marks_sensitive_content():
    raw = _zip(
        {
            "LTM-Day32.md": """# LTM Day32 · 2026-07-17
## 技术进展
- **部署完成**：远程服务已上线并通过检查。
- **后续工作**：保留数据库备份并观察召回质量。
## 健康记录
- 今天需要去医院复诊，并继续遵循医生建议。
"""
        }
    )

    parsed = parse_ltm_archive(raw)
    assert parsed["preview"]["documents"] == 1
    assert parsed["preview"]["memories"] == 2
    assert parsed["preview"]["categories"] == {"health": 1, "worklog": 1}
    assert parsed["preview"]["privacy"] == {"personal": 1, "sensitive": 1}
    technical = next(memory for memory in parsed["memories"] if memory["category"] == "worklog")
    assert "部署完成" in technical["content"]
    assert "后续工作" in technical["content"]
    health = next(memory for memory in parsed["memories"] if memory["category"] == "health")
    assert health["surface_allowed"] is False
    assert health["created_at"].isoformat().startswith("2026-07-17")


def test_relationship_structure_outranks_technical_words():
    raw = _zip(
        {
            "LTM-baseline.md": """# 甲 × 乙 · 长期记忆档案
## 0. 最重要的（先读这一段）
- 甲与乙使用这份档案维持跨窗口连续性。
""",
            "LTM-Day33.md": """# LTM Day33 · 2026-07-18
## 乙的状态更新
- 今天讨论了技术方案，但这段记录的核心是双方的关系变化和边界确认。
"""
        }
    )
    parsed = parse_ltm_archive(raw)
    memory = next(memory for memory in parsed["memories"] if memory["title"] == "乙的状态更新")
    assert memory["category"] == "relationship_moment"
    assert memory["metadata"]["category_reason"] == "section:participant-state"
    assert memory["metadata"]["category_review"] is False


def test_ambiguous_ltm_section_falls_back_to_reviewable_episode():
    raw = _zip(
        {
            "LTM-Day33.md": """# LTM Day33 · 2026-07-18
## 技术状态更新
- 今天重新整理了一些东西，之后再决定它属于哪条线。
"""
        }
    )
    parsed = parse_ltm_archive(raw)
    memory = parsed["memories"][0]
    assert memory["category"] == "episode"
    assert memory["metadata"]["category_review"] is True
    assert parsed["preview"]["category_review"] == 1


def test_ltm_bold_subsection_keeps_all_bullets_and_gets_entity_tags():
    raw = _zip(
        {
            "LTM-Day24.md": """# LTM Day24 · 2026-07-09
## 新增/更新信息
**离婚计划（当前版本）**
- 20号左右请假回湖南接孩子。
- 月底合同到期不续签，再起诉离婚。
- 争取抚养权并让律师修改离婚协议。

**工作现状**
- 职业：韵达快递结算文员。
- 如果公司裁员，需要考虑N+1赔偿。

**回湖南规划**
- 回湖南后先安顿，再面试新的工作。
"""
        }
    )
    parsed = parse_ltm_archive(raw)
    assert parsed["preview"]["memories"] == 3
    divorce = next(memory for memory in parsed["memories"] if memory["title"] == "离婚计划（当前版本）")
    assert divorce["category"] == "legal"
    assert all(term in divorce["content"] for term in ("20号", "合同", "抚养权"))
    assert "新增/更新信息" not in divorce["title"]
    work = next(memory for memory in parsed["memories"] if memory["title"] == "工作现状")
    assert work["category"] == "worklog"
    assert "韵达快递" in work["tags"]
    assert "工作" in work["tags"]
    assert work["metadata"]["classification_version"] == "structure-v3"


def test_zip_path_traversal_is_rejected():
    raw = _zip({"../outside.md": "# no"})
    with pytest.raises(ValueError, match="unsafe path"):
        parse_ltm_archive(raw)
