#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def _pick(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _doc_id(row: dict[str, Any]) -> str:
    return str(_pick(row, "id", "doc_id", "document_id", "_id", "pid", "corpus_id"))


def _query_id(row: dict[str, Any]) -> str:
    return str(_pick(row, "id", "query_id", "qid", "_id"))


def _query_text(row: dict[str, Any]) -> str:
    return str(_pick(row, "query", "question", "prompt", "text"))


def _doc_text(row: dict[str, Any]) -> str:
    return str(_pick(row, "text", "content", "passage", "document", "body"))


def _doc_title(row: dict[str, Any]) -> str:
    return str(_pick(row, "title", "name", default=""))


def _answers(row: dict[str, Any]) -> list[str]:
    value = _pick(row, "answers", "answer", "target", "gold_answer", default=[])
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _load_qrels(path: str | Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    qrels_path = Path(path).expanduser()
    if qrels_path.suffix.lower() == ".jsonl":
        for row in _read_jsonl(qrels_path):
            qid = str(_pick(row, "query_id", "qid", "id"))
            doc_id = str(_pick(row, "doc_id", "document_id", "corpus_id", "pid"))
            score = _pick(row, "score", "relevance", "label", default=1)
            try:
                positive = float(score) > 0
            except (TypeError, ValueError):
                positive = bool(score)
            if qid and doc_id and positive:
                qrels.setdefault(qid, []).append(doc_id)
        return qrels

    with qrels_path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t, ")
        reader = csv.reader(handle, dialect)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0].lower() in {"qid", "query_id"}:
                continue
            if len(row) >= 4:
                qid, doc_id, score = row[0], row[2], row[3]
            elif len(row) >= 3:
                qid, doc_id, score = row[0], row[1], row[2]
            else:
                raise ValueError(f"Invalid qrels row in {qrels_path}: {row}")
            try:
                positive = float(score) > 0
            except (TypeError, ValueError):
                positive = bool(score)
            if positive:
                qrels.setdefault(str(qid), []).append(str(doc_id))
    return qrels


def _load_run(path: str | Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    run_path = Path(path).expanduser()
    if run_path.suffix.lower() == ".jsonl":
        for row in _read_jsonl(run_path):
            qid = str(_pick(row, "query_id", "qid", "id"))
            doc_ids = _pick(row, "doc_ids", "document_ids", "candidates", "ranking", default=[])
            if isinstance(doc_ids, list):
                ids = [str(item if not isinstance(item, dict) else _doc_id(item)) for item in doc_ids]
            else:
                ids = [str(doc_ids)]
            run.setdefault(qid, []).extend(ids)
        return run

    with run_path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t, ")
        reader = csv.reader(handle, dialect)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0].lower() in {"qid", "query_id"}:
                continue
            if len(row) >= 3 and row[1].upper() == "Q0":
                qid, doc_id = row[0], row[2]
            elif len(row) >= 2:
                qid, doc_id = row[0], row[1]
            else:
                raise ValueError(f"Invalid run row in {run_path}: {row}")
            run.setdefault(str(qid), []).append(str(doc_id))
    return run


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize query/corpus/qrels retrieval data into reasoning-rag-env JSONL."
    )
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--qrels", required=True, help="JSONL, TSV, CSV, TREC qrels, or BEIR qrels.")
    parser.add_argument("--run", help="Optional candidate ranking/run file. Positives are always included.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--task", default="rag")
    parser.add_argument("--max-documents", type=int, default=48)
    parser.add_argument("--max-document-chars", type=int, default=4000)
    parser.add_argument("--random-negatives", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    queries = {_query_id(row): row for row in _read_jsonl(args.queries_jsonl)}
    corpus_rows = _read_jsonl(args.corpus_jsonl)
    corpus = {_doc_id(row): row for row in corpus_rows}
    qrels = _load_qrels(args.qrels)
    run = _load_run(args.run) if args.run else {}
    all_doc_ids = list(corpus.keys())
    rng = random.Random(args.seed)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for qid, query_row in queries.items():
            gold_ids = _dedupe([doc_id for doc_id in qrels.get(qid, []) if doc_id in corpus])
            if not gold_ids:
                skipped += 1
                continue
            candidate_ids = _dedupe(gold_ids + [doc_id for doc_id in run.get(qid, []) if doc_id in corpus])
            if args.random_negatives > 0:
                forbidden = set(candidate_ids)
                pool = [doc_id for doc_id in all_doc_ids if doc_id not in forbidden]
                sample_size = min(args.random_negatives, len(pool))
                candidate_ids.extend(rng.sample(pool, sample_size))
            candidate_ids = _dedupe(candidate_ids)[: args.max_documents]
            documents = []
            for doc_id in candidate_ids:
                doc = corpus[doc_id]
                documents.append(
                    {
                        "id": doc_id,
                        "title": _doc_title(doc),
                        "text": _doc_text(doc)[: args.max_document_chars],
                        "relevant": doc_id in gold_ids,
                    }
                )
            row = {
                "id": qid,
                "benchmark": args.benchmark,
                "task": args.task,
                "query": _query_text(query_row),
                "answers": _answers(query_row),
                "gold_doc_ids": gold_ids,
                "documents": documents,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote={written} skipped_without_gold={skipped} output={output_path}")


if __name__ == "__main__":
    main()
