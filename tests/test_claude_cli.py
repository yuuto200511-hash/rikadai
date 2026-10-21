import json
import subprocess

import pytest

from src.llm import claude_cli


@pytest.fixture(autouse=True)
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_cli, "LOG_PATH", tmp_path / "llm.jsonl")


def fake_run(outputs):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        out = outputs[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        rc, stdout = out
        return subprocess.CompletedProcess(cmd, rc, stdout, "boom")

    run.calls = calls
    return run


def ok(text):
    return (0, json.dumps({"result": text, "is_error": False, "total_cost_usd": 0.01}))


def test_ask_passes_prompt_via_stdin_and_logs(monkeypatch):
    run = fake_run([ok("hello")])
    monkeypatch.setattr(subprocess, "run", run)
    assert claude_cli.ask("hi", system="sys") == "hello"
    cmd, kw = run.calls[0]
    assert kw["input"] == "hi" and cmd[cmd.index("--system-prompt") + 1] == "sys"
    rec = json.loads(claude_cli.LOG_PATH.read_text().splitlines()[0])
    assert rec["prompt"] == "hi" and rec["response"] == "hello"


@pytest.mark.parametrize(
    "out",
    [
        subprocess.TimeoutExpired("claude", 1),
        (1, ""),
        (0, ""),
        (0, json.dumps({"result": "", "is_error": False})),
        (0, json.dumps({"result": "rate limited", "is_error": True})),
    ],
)
def test_ask_errors(monkeypatch, out):
    monkeypatch.setattr(subprocess, "run", fake_run([out]))
    with pytest.raises(claude_cli.ClaudeError):
        claude_cli.ask("hi")
    assert "error" in json.loads(claude_cli.LOG_PATH.read_text().splitlines()[0])


def test_ask_json_retries_once(monkeypatch):
    run = fake_run([ok("not json"), ok('```json\n{"a": 1}\n```')])
    monkeypatch.setattr(subprocess, "run", run)
    assert claude_cli.ask_json("q", "{a: int}") == {"a": 1}
    assert len(run.calls) == 2


def test_ask_json_fails_after_retry(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run([ok("nope"), ok("still nope")]))
    with pytest.raises(claude_cli.ClaudeError):
        claude_cli.ask_json("q", "{}")


@pytest.mark.parametrize("effort", ["low", ""])
def test_build_cmd_effort_and_no_settings(monkeypatch, effort):
    monkeypatch.setattr(claude_cli, "CLAUDE_EFFORT", effort)
    cmd = claude_cli._build_cmd("sys", "m")
    assert cmd[cmd.index("--setting-sources") + 1] == "" and "--strict-mcp-config" in cmd
    assert cmd[-2:] == ["--tools", ""]
    if effort:
        assert cmd[cmd.index("--effort") + 1] == "low"
    else:
        assert "--effort" not in cmd


@pytest.mark.parametrize(
    "text, expected",
    [
        ("連絡はtaro.yamada+lab@ed.tus.ac.jpまで", "連絡は[伏字]まで"),
        ("学籍番号6322001です", "学籍番号[伏字]です"),
        ("学籍番号６３２２００１です", "学籍番号[伏字]です"),
        # 7桁でない数字・日付・時刻はそのまま
        ("123456 と 12345678、2026-10-20 12:30", "123456 と 12345678、2026-10-20 12:30"),
    ],
)
def test_redact(text, expected):
    assert claude_cli.redact(text) == expected


def test_log_redacts_but_prompt_is_unchanged(monkeypatch):
    run = fake_run([ok("a@example.com さん、6322001 ですね")])
    monkeypatch.setattr(subprocess, "run", run)
    prompt = "6322001 の a@example.com です"
    assert claude_cli.ask(prompt) == "a@example.com さん、6322001 ですね"
    assert run.calls[0][1]["input"] == prompt  # AI に渡す内容は変えない
    rec = json.loads(claude_cli.LOG_PATH.read_text())
    assert rec["prompt"] == "[伏字] の [伏字] です"
    assert rec["response"] == "[伏字] さん、[伏字] ですね"
    assert rec["cost_usd"] == 0.01
