#!/usr/bin/env python3
"""Build the small runtime knowledge index from synced JSON documents."""

from __future__ import annotations

import json
import os
from pathlib import Path

APP_DIR = Path("/home/elf/Projects/voice_assistant")
SOURCE_DIR = APP_DIR / "custom_knowledge"
TARGET = APP_DIR / "knowledge.json"


def main() -> int:
    index = {}
    for source in sorted(SOURCE_DIR.glob("*.json")):
        try:
            entry = json.loads(source.read_text(encoding="utf-8"))
            title = str(entry["title"]).strip()
            content = str(entry["content"]).strip()
            keywords = [str(keyword).strip() for keyword in entry.get("keywords", []) if str(keyword).strip()]
            if title and content and keywords:
                index[f"custom:{entry['id']}"] = {"keywords": keywords, "content": content}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    temporary = TARGET.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
