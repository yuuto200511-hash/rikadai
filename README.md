# キャンパス・スポットナビ（仮称）

設計方針は [DESIGN.md](DESIGN.md)。Claude は `claude -p`（Claude Code CLI）経由で呼び出す（`src/llm/claude_cli.py`）。

## セットアップと起動

```bash
uv sync
uv run python -m src.ingest.survey          # data/survey.xlsx（spots・reservations シート）を取込。除外行と空欄の一覧を表示
uv run python -m src.db.vector_store        # スポット（場所＋備考）/ 口コミをベクトル DB に登録
uv run python -m src.ingest.notice          # data/notices/*.txt を判定・抽出（confirmed=false で保存）。現在はデモ用の架空メール10件
uv run streamlit run app.py                 # 利用者向け画面。サイドバーの「管理」（pages/）でお知らせを確認済みにすると検索に反映
```

仮データ（10スポット）は `uv run python scripts/make_sample_data.py` で `data/sample/survey_sample.xlsx` に生成できる
（`data/survey.xlsx` は上書きしない）。取り込むときは `SPOT_DB_PATH` / `SPOT_CHROMA_PATH` で別の DB を指定する。

## テスト・評価

```bash
uv run pytest
RUN_LLM_TESTS=1 uv run pytest tests/test_typos.py   # 誤字入りの質問の条件抽出（Claude を呼ぶ）
uv run python eval/run_eval.py              # 結果は eval/results/*.csv（questions.yaml は現在まだ仮データ用）
```

## 環境変数

| 変数 | 既定値 | 用途 |
|---|---|---|
| `CLAUDE_MODEL` | `claude-sonnet-5` | `claude -p --model` に渡すモデル |
| `CLAUDE_EFFORT` | `low` | `claude -p --effort` に渡す思考の量（空にすると付けない＝モデルの既定） |
| `CLAUDE_BIN` | `claude` | CLI のパス |
| `SPOT_DB_PATH` | `data/spots.db` | SQLite |
| `SPOT_CHROMA_PATH` | `data/chroma` | Chroma |
| `SPOT_ROOMS_PATH` | `data/rooms.csv` | 実在する教室の一覧 |
