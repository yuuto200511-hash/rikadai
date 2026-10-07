"""Claude Code CLI (headless `claude -p`) 経由で Claude を呼び出す。

ctf-agent (backend/agents/claude_code_coordinator.py) の `_build_cmd` / `_run_claude` から
呼び出し・出力の受け取り・エラー処理の部分だけを切り出したもの。
本プロジェクトで Claude を呼ぶのはこのモジュールだけにすること。

ctf-agent からの変更点:
- asyncio → subprocess.run（同期）
- プロンプトは argv ではなく stdin で渡す（長い検索結果を含むため）
- ツールは `--tools ""` で全て無効化（テキスト応答のみ）
- ユーザー・プロジェクトの設定と MCP サーバーは読み込まない（起動を速くするため）
- セッション継続（--resume）・コスト集計・ツール呼び出しループは持ち込まない
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
# 思考の量（low / medium / high ...）。空にすると --effort を付けない（モデルの既定）
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "low")
LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "llm.jsonl"

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ClaudeError(RuntimeError):
    """Claude 呼び出しの失敗（タイムアウト・非ゼロ終了・空出力・JSON 不正）。"""


def _build_cmd(system: str | None, model: str) -> list[str]:
    cmd = [
        CLAUDE_BIN,
        "-p",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--model",
        model,
        # 設定ファイル（hooks・プラグイン等）と MCP サーバーを読み込まない
        "--setting-sources",
        "",
        "--strict-mcp-config",
    ]
    if CLAUDE_EFFORT:
        cmd += ["--effort", CLAUDE_EFFORT]
    if system:
        cmd += ["--system-prompt", system]
    # --tools は可変長引数なので最後に置く（プロンプトは stdin で渡す）
    cmd += ["--tools", ""]
    return cmd


def _log(record: dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _run_claude(prompt: str, system: str | None, timeout: int, model: str) -> dict[str, Any]:
    """`claude -p` を実行し、JSON 応答 dict を返す。"""
    cmd = _build_cmd(system, model)
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ClaudeError(f"`claude` subprocess timed out after {timeout}s") from None
    except FileNotFoundError:
        raise ClaudeError(f"`{CLAUDE_BIN}` コマンドが見つかりません") from None

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout)[:500]
        raise ClaudeError(f"`claude` exited {proc.returncode}: {err}")

    raw = proc.stdout.strip()
    if not raw:
        raise ClaudeError("`claude` returned empty output")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClaudeError(f"`claude` JSON parse error: {exc}\nraw={raw[:400]}") from exc
    if data.get("is_error"):
        raise ClaudeError(f"`claude` reported error: {str(data.get('result'))[:400]}")
    return data


def ask(
    prompt: str,
    system: str | None = None,
    timeout: int = 120,
    *,
    model: str | None = None,
) -> str:
    """Claude にプロンプトを渡し、テキスト応答を返す"""
    model = model or DEFAULT_MODEL
    t0 = time.monotonic()
    record: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "effort": CLAUDE_EFFORT,
        "system": system,
        "prompt": prompt,
    }
    try:
        data = _run_claude(prompt, system, timeout, model)
        result = str(data.get("result") or "").strip()
        if not result:
            raise ClaudeError("`claude` returned empty result")
        record.update(
            response=result,
            cost_usd=data.get("total_cost_usd"),
            duration_s=round(time.monotonic() - t0, 2),
        )
        return result
    except ClaudeError as exc:
        record.update(error=str(exc), duration_s=round(time.monotonic() - t0, 2))
        raise
    finally:
        _log(record)


def _parse_json(text: str) -> dict:
    m = _JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else text
    if not m:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("JSON object ではありません")
    return obj


def ask_json(prompt: str, schema_hint: str, system: str | None = None) -> dict:
    """JSON 形式の応答を要求し、パースして返す。
    パース失敗時は1回だけ再試行し、それでも失敗したら例外を投げる"""
    full_prompt = (
        f"{prompt}\n\n"
        "## 出力形式\n"
        "次の形式の JSON オブジェクトのみを出力してください。説明文やコードフェンスは不要です。\n"
        f"{schema_hint}"
    )
    last_err: Exception | None = None
    for _ in range(2):
        text = ask(full_prompt, system=system)
        try:
            return _parse_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_err = exc
    raise ClaudeError(f"JSON 応答のパースに失敗しました: {last_err}")
