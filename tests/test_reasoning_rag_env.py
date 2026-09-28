from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reasoning_rag_env import load_environment
from reasoning_rag_env.env import (
    extract_doc_ids,
    score_answer,
    score_retrieval,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_reasoning_rag_env_builds_prompt_and_scores_doc_ids(tmp_path: Path) -> None:
    jsonl = tmp_path / "rag.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "id": "q1",
                "benchmark": "bright",
                "task": "rag",
                "query": "Which document proves the theorem?",
                "answer": "Doc A proves it.",
                "gold_doc_ids": ["d1"],
                "documents": [
                    {"id": "d1", "title": "Proof", "text": "The theorem follows by induction."},
                    {"id": "d2", "title": "Distractor", "text": "Unrelated background."},
                ],
            }
        ],
    )

    env = load_environment(
        local_jsonl_path=str(jsonl),
        num_train_examples=1,
        num_eval_examples=0,
        max_documents=8,
        max_document_chars=1000,
    )
    dataset = env.get_dataset()

    assert len(dataset) == 1
    assert dataset[0]["src_id"] == "q1"
    assert "[DOC d1]" in dataset[0]["prompt"][1]["content"]
    completion = [{"role": "assistant", "content": "<answer>Doc A proves it.</answer>\n<doc_ids>d1</doc_ids>"}]
    assert score_retrieval(completion, dataset[0]["info"], top_k=5) == 1.0
    assert score_answer(completion, dataset[0]["answer"], dataset[0]["info"]) == 1.0


def test_reasoning_rag_env_requires_real_supervision(tmp_path: Path) -> None:
    jsonl = tmp_path / "unsupervised.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "id": "q1",
                "query": "No labels are provided.",
                "documents": [{"id": "d1", "text": "Context only."}],
            }
        ],
    )

    env = load_environment(local_jsonl_path=str(jsonl), num_eval_examples=0)
    with pytest.raises(ValueError, match="neither gold_doc_ids nor answers"):
        env.get_dataset()


def test_reasoning_rag_multiple_choice_answer_reward(tmp_path: Path) -> None:
    jsonl = tmp_path / "mcq.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "id": "q1",
                "task": "multiple_choice",
                "query": "Which option is supported?",
                "answer": "B",
                "answer_choices": {"A": "Wrong", "B": "Correct"},
                "documents": [{"id": "d1", "text": "Correct is supported.", "relevant": True}],
            }
        ],
    )

    env = load_environment(local_jsonl_path=str(jsonl), num_eval_examples=0)
    row = env.get_dataset()[0]
    completion = [{"role": "assistant", "content": "<answer>Correct</answer>\n<doc_ids>d1</doc_ids>"}]

    assert score_answer(completion, row["answer"], row["info"]) == 1.0
    assert score_retrieval(completion, row["info"], top_k=5) == 1.0


def test_extract_doc_ids_tolerates_common_formats() -> None:
    assert extract_doc_ids("<doc_ids>d-1, case.2</doc_ids>") == ["d-1", "case.2"]
    assert extract_doc_ids('{"citations": ["a", "b"]}') == ["a", "b"]
    assert extract_doc_ids("See [DOC statute-7] for the rule.") == ["statute-7"]


def test_build_reasoning_rag_jsonl_materializes_qrels(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    qrels = tmp_path / "qrels.tsv"
    out = tmp_path / "out.jsonl"
    _write_jsonl(
        queries,
        [
            {"id": "q1", "query": "Which case controls?", "answer": "Case A"},
            {"id": "q2", "query": "No local gold document."},
        ],
    )
    _write_jsonl(
        corpus,
        [
            {"id": "d1", "title": "Case A", "text": "Controlling case."},
            {"id": "d2", "title": "Case B", "text": "Distractor."},
        ],
    )
    qrels.write_text("q1\t0\td1\t1\nq2\t0\tmissing\t1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_reasoning_rag_jsonl.py",
            "--queries-jsonl",
            str(queries),
            "--corpus-jsonl",
            str(corpus),
            "--qrels",
            str(qrels),
            "--benchmark",
            "legal-retrieval",
            "--output",
            str(out),
            "--random-negatives",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert "wrote=1" in result.stdout
    assert rows[0]["id"] == "q1"
    assert rows[0]["gold_doc_ids"] == ["d1"]
    assert rows[0]["documents"][0]["id"] == "d1"
