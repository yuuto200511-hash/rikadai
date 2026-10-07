"""プロンプトは src/prompts/*.txt で管理する。置換は string.Template（$name）。"""

from __future__ import annotations

from pathlib import Path
from string import Template

_DIR = Path(__file__).parent


def load(name: str, **values: object) -> str:
    text = (_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return Template(text).substitute(**{k: str(v) for k, v in values.items()})
