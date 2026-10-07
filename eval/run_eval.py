"""想定質問で回答精度を評価する: uv run python eval/run_eval.py [--skip-judge] [--skip-notices]

指標
- 正答率: normal は正解スポットが上位3件に含まれる / no_match は該当なし扱い（条件緩和 or 0件）
         / unsupported は対応外判定
- ハルシネーション率: 回答に DB にない・根拠に渡していないスポットを含む（ルール判定）
                     または根拠にない主張を含む（LLM 判定）回答の割合
- お知らせ関連判定の正答率
- 応答時間: 1問の回答にかかった秒数（LLM 判定は含まない）の平均・最大
結果は eval/results/ に CSV で出力する。
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import prompts  # noqa: E402
from src.db import repository  # noqa: E402
from src.ingest import notice as notice_ingest  # noqa: E402
from src.llm import claude_cli  # noqa: E402
from src.rag.answer import Answer, answer_question, build_prompt  # noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
SPOT_ID_RE = re.compile(r"\bS\d{3}\b")

KIND_TO_BEHAVIOR = {
    "normal": "normal",
    "relaxed": "no_match",
    "no_result": "no_match",
    "unsupported": "unsupported",
    "clarify": "clarify",
    "unsurveyed": "unsupported",
    "notice": "notice",
    "chat": "chat",
    "error": "error",
}


def rule_check(ans: Answer, all_spots: list[dict]) -> list[str]:
    """DB に存在しない / 根拠として渡していないスポットへの言及を検出する。"""
    problems = []
    known = {s["spot_id"]: s for s in all_spots}
    given = set()
    if ans.retrieval:
        given = {s["spot_id"] for s in ans.retrieval.spots}
        given |= {e["spot"]["spot_id"] for e in ans.retrieval.excluded}
    for sid in set(SPOT_ID_RE.findall(ans.text)):
        if sid not in known:
            problems.append(f"DBにないスポットID {sid}")
        elif sid not in given:
            problems.append(f"根拠外のスポットID {sid}")
    for sid, s in known.items():
        if sid not in given and s["area_name"] in ans.text and s["area_name"] not in ans.context:
            problems.append(f"根拠外のスポット {s['building']} {s['area_name']}")
    return problems


def judge(question: str, ans: Answer, today: date | None) -> tuple[bool, list[str]]:
    # 返事は LLM が書く（失敗時の定型文は context が空）。根拠は返事を書かせたときに渡したデータ
    context = ans.context or (build_prompt(question, ans.retrieval, today) if ans.retrieval else "")
    if not context:
        return False, []  # 定型文の回答は LLM 判定不要
    res = claude_cli.ask_json(
        prompts.load("eval_judge", context=context, answer=ans.text),
        prompts.load("eval_judge_schema"),
        system=prompts.load("eval_judge_system"),
    )
    claims = [str(c) for c in res.get("unsupported_claims") or []]
    return bool(res.get("hallucinated")) or bool(claims), claims


def eval_questions(conn, skip_judge: bool) -> tuple[list[dict], dict]:
    spec = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text(encoding="utf-8"))
    today = spec.get("today")
    today = date.fromisoformat(str(today)) if today else None
    all_spots = repository.list_spots(conn)
    rows = []
    for q in spec["questions"]:
        print(f"[{q['id']}] {q['question']}", flush=True)
        t0 = time.monotonic()
        ans = answer_question(q["question"], conn, today=today)
        seconds = round(time.monotonic() - t0, 1)
        behavior = KIND_TO_BEHAVIOR[ans.kind]
        answers = q.get("answers") or []
        if q["expected"] == "normal":
            correct = behavior == "normal" and any(a in ans.spot_ids for a in answers)
        else:
            correct = behavior == q["expected"]
        problems = rule_check(ans, all_spots) if ans.kind != "error" else []
        judged, claims = (False, [])
        if not skip_judge and ans.kind != "error":
            try:
                judged, claims = judge(q["question"], ans, today)
            except claude_cli.ClaudeError as e:
                claims = [f"(判定失敗: {e})"]
        hallucinated = bool(problems) or judged
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected": q["expected"],
                "answers": " ".join(answers),
                "kind": ans.kind,
                "behavior": behavior,
                "top3": " ".join(ans.spot_ids),
                "correct": int(correct),
                "hallucinated": int(hallucinated) if ans.kind != "error" else "",
                "rule_problems": " / ".join(problems),
                "judge_claims": " / ".join(claims),
                "seconds": seconds,
                "conditions": str(ans.conditions),
                "answer": ans.text,
            }
        )
        print(f"    -> {ans.kind} {ans.spot_ids} correct={correct} halluc={hallucinated} {seconds}s")
    n = len(rows)
    answered = [r for r in rows if r["kind"] != "error"]
    summary = {
        "questions": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0.0,
        "hallucination_rate": (
            sum(r["hallucinated"] for r in answered) / len(answered) if answered else 0.0
        ),
        "errors": n - len(answered),
        "seconds_mean": statistics.mean(r["seconds"] for r in rows) if rows else 0.0,
        "seconds_max": max((r["seconds"] for r in rows), default=0.0),
    }
    for exp in ("normal", "no_match", "unsupported"):
        sub = [r for r in rows if r["expected"] == exp]
        if sub:
            summary[f"accuracy_{exp}"] = sum(r["correct"] for r in sub) / len(sub)
    return rows, summary


def eval_notices() -> tuple[list[dict], dict]:
    labels = yaml.safe_load((EVAL_DIR / "notice_labels.yaml").read_text(encoding="utf-8"))
    rows = []
    for raw in notice_ingest.DirectorySource().fetch():
        if raw.source_id not in labels:
            continue
        try:
            pred = notice_ingest.judge_relevance(notice_ingest.strip_demo_mark(raw.body))
            err = ""
        except claude_cli.ClaudeError as e:
            pred, err = None, str(e)
        label = bool(labels[raw.source_id])
        rows.append(
            {"file": raw.source_id, "label": int(label),
             "pred": "" if pred is None else int(pred), "correct": int(pred == label), "error": err}
        )  # fmt: skip
        print(f"[notice] {raw.source_id}: label={label} pred={pred}")
    n = len(rows)
    return rows, {"notices": n, "notice_accuracy": sum(r["correct"] for r in rows) / n if n else 0}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-judge", action="store_true", help="LLM によるハルシネーション判定を省略")
    ap.add_argument("--skip-notices", action="store_true", help="お知らせ関連判定の評価を省略")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conn = repository.connect()

    rows, summary = eval_questions(conn, args.skip_judge)
    write_csv(RESULTS_DIR / f"{stamp}_questions.csv", rows)
    if not args.skip_notices:
        n_rows, n_summary = eval_notices()
        write_csv(RESULTS_DIR / f"{stamp}_notices.csv", n_rows)
        summary.update(n_summary)

    summary_rows = [{"metric": k, "value": round(v, 4) if isinstance(v, float) else v}
                    for k, v in summary.items()]  # fmt: skip
    write_csv(RESULTS_DIR / f"{stamp}_summary.csv", summary_rows)
    print("\n=== summary ===")
    for r in summary_rows:
        print(f"{r['metric']}: {r['value']}")
    print(f"results: {RESULTS_DIR}/{stamp}_*.csv")


if __name__ == "__main__":
    main()
