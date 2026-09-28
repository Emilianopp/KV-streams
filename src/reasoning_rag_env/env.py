from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.types import Messages

DEFAULT_SYSTEM_PROMPT = """\
You are a reasoning-focused retrieval and RAG model.

You will receive one query and a set of candidate documents. Use only the
provided documents. Reason carefully, select the most relevant evidence, and
return your answer in this exact format:

<answer>your final answer</answer>
<doc_ids>doc_id_1, doc_id_2, ...</doc_ids>

The doc_ids field should list the smallest set of provided document IDs needed
to justify the answer, ordered by relevance. If the task is retrieval-only,
still provide the best answer you can, but the doc_ids field is mandatory.
"""

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_DOC_IDS_RE = re.compile(r"<doc_ids>\s*(.*?)\s*</doc_ids>", re.IGNORECASE | re.DOTALL)
_JSON_IDS_RE = re.compile(
    r'"(?:doc_ids|document_ids|citations|evidence_ids)"\s*:\s*\[(.*?)\]',
    re.IGNORECASE | re.DOTALL,
)
_DOC_REF_RE = re.compile(r"\[DOC\s+([^\]\n]+?)\]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0:1] in "[{":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return list(parsed.keys())
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                continue
            text_attr = getattr(part, "text", None)
            if isinstance(text_attr, str):
                chunks.append(text_attr)
        return "\n".join(chunks)
    return str(content)


def completion_text(completion: Messages | str) -> str:
    if isinstance(completion, str):
        return completion
    for message in reversed(completion):
        if isinstance(message, Mapping):
            role = message.get("role")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", "")
        if role == "assistant":
            return _content_to_text(content)
    return ""


def normalize_text(text: Any) -> str:
    return " ".join(str(text).lower().strip().split())


def normalize_choice(text: Any) -> str:
    normalized = normalize_text(text)
    if len(normalized) == 1 and normalized.isalpha():
        return normalized.upper()
    match = re.match(r"^\(?([a-zA-Z])\)?[.)\s:-]", str(text).strip())
    if match:
        return match.group(1).upper()
    return normalized


def extract_answer_text(text: str) -> str:
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _split_doc_id_blob(blob: str) -> list[str]:
    out: list[str] = []
    for raw in re.split(r"[,;\n]", blob):
        token = raw.strip().strip("\"'`[](){} ")
        if token:
            out.append(token)
    return out


def extract_doc_ids(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add_many(values: Iterable[Any]) -> None:
        for value in values:
            token = str(value).strip().strip("\"'`[](){} ")
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)

    match = _DOC_IDS_RE.search(text)
    if match:
        add_many(_split_doc_id_blob(match.group(1)))

    for json_match in _JSON_IDS_RE.finditer(text):
        add_many(_split_doc_id_blob(json_match.group(1)))

    add_many(_DOC_REF_RE.findall(text))
    return out


class ReasoningRagParser(vf.Parser):
    def parse(self, text: str) -> str:
        return extract_answer_text(text)

    def parse_answer(self, completion: Messages) -> str | None:
        text = completion_text(completion)
        if not text:
            return None
        return extract_answer_text(text)

    def parse_doc_ids(self, completion: Messages | str) -> list[str]:
        return extract_doc_ids(completion_text(completion))


def _token_f1(prediction: str, references: list[str]) -> float:
    pred_tokens = _TOKEN_RE.findall(prediction.lower())
    if not pred_tokens or not references:
        return 0.0
    best = 0.0
    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for reference in references:
        ref_tokens = _TOKEN_RE.findall(reference.lower())
        if not ref_tokens:
            continue
        ref_counts: dict[str, int] = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        score = 2.0 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def _info_dict(info: Any) -> dict[str, Any]:
    if isinstance(info, dict):
        return info
    if isinstance(info, str) and info.strip():
        try:
            parsed = json.loads(info)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _answer_references(info: dict[str, Any], answer: Any) -> list[str]:
    refs = _string_list(info.get("answers") or info.get("reference_answers"))
    if not refs and answer not in (None, ""):
        refs = _string_list(answer)
    return refs


def _answer_type(info: dict[str, Any]) -> str:
    task = str(info.get("task") or "").lower()
    explicit = str(info.get("answer_type") or "").lower()
    if explicit:
        return explicit
    if task in {"multiple_choice", "mcq", "yes_no", "binary"}:
        return task
    choices = info.get("answer_choices") or info.get("choices")
    if choices:
        return "multiple_choice"
    refs = [normalize_text(ref) for ref in _string_list(info.get("answers"))]
    if refs and all(ref in {"yes", "no", "true", "false"} for ref in refs):
        return "yes_no"
    return "free_form"


def score_answer(completion: Messages | str, answer: Any, info: Any) -> float:
    info_dict = _info_dict(info)
    refs = _answer_references(info_dict, answer)
    if not refs:
        return 0.0

    pred = extract_answer_text(completion_text(completion))
    pred_norm = normalize_text(pred)
    answer_type = _answer_type(info_dict)

    if answer_type in {"multiple_choice", "mcq"}:
        pred_choice = normalize_choice(pred)
        ref_choices = {normalize_choice(ref) for ref in refs}
        choices = info_dict.get("answer_choices") or info_dict.get("choices") or {}
        if isinstance(choices, Mapping):
            for key, value in choices.items():
                if normalize_text(value) == pred_norm:
                    pred_choice = str(key).upper()
                for ref in refs:
                    if normalize_text(value) == normalize_text(ref):
                        ref_choices.add(str(key).upper())
        return 1.0 if pred_choice in ref_choices else 0.0

    if answer_type in {"yes_no", "binary"}:
        def yn(value: str) -> str:
            normalized = normalize_text(value)
            if normalized.startswith("yes") or normalized == "true":
                return "yes"
            if normalized.startswith("no") or normalized == "false":
                return "no"
            return normalized

        pred_yn = yn(pred)
        return 1.0 if any(pred_yn == yn(ref) for ref in refs) else 0.0

    exact_refs = {normalize_text(ref) for ref in refs}
    if pred_norm in exact_refs:
        return 1.0
    return _token_f1(pred, refs)


def score_retrieval(completion: Messages | str, info: Any, *, top_k: int = 5) -> float:
    info_dict = _info_dict(info)
    gold_ids = _string_list(info_dict.get("gold_doc_ids") or info_dict.get("relevant_doc_ids"))
    if not gold_ids:
        return 0.0
    predicted = extract_doc_ids(completion_text(completion))
    if not predicted:
        return 0.0
    k = max(1, top_k)
    pred_set = set(predicted[:k])
    gold_set = set(gold_ids)
    return len(pred_set & gold_set) / len(gold_set)


class ReasoningRagRubric(vf.Rubric):
    def __init__(
        self,
        *,
        parser: ReasoningRagParser,
        reward_top_k: int = 5,
        answer_weight: float = 0.7,
        retrieval_weight: float = 0.3,
    ) -> None:
        super().__init__(parser=parser)
        self.reward_top_k = reward_top_k
        self.answer_weight = answer_weight
        self.retrieval_weight = retrieval_weight
        self.add_reward_func(self.combined_reward_func)
        self.add_metric(self.answer_score_func)
        self.add_metric(self.retrieval_score_func)
        self.add_metric(self.format_score_func)
        self.add_metric(self.has_answer_label_func)
        self.add_metric(self.has_gold_docs_func)

    def answer_score_func(self, completion: Messages | str, answer: Any, info: Any, **kwargs: Any) -> float:
        return score_answer(completion, answer, info)

    def retrieval_score_func(self, completion: Messages | str, info: Any, **kwargs: Any) -> float:
        return score_retrieval(completion, info, top_k=self.reward_top_k)

    def format_score_func(self, completion: Messages | str, **kwargs: Any) -> float:
        text = completion_text(completion)
        has_answer = bool(_ANSWER_RE.search(text))
        has_doc_ids = bool(extract_doc_ids(text))
        return 1.0 if has_answer and has_doc_ids else 0.0

    def has_answer_label_func(self, answer: Any, info: Any, **kwargs: Any) -> float:
        return 1.0 if _answer_references(_info_dict(info), answer) else 0.0

    def has_gold_docs_func(self, info: Any, **kwargs: Any) -> float:
        info_dict = _info_dict(info)
        return 1.0 if _string_list(info_dict.get("gold_doc_ids") or info_dict.get("relevant_doc_ids")) else 0.0

    def combined_reward_func(self, completion: Messages | str, answer: Any, info: Any, **kwargs: Any) -> float:
        info_dict = _info_dict(info)
        has_answer = bool(_answer_references(info_dict, answer))
        has_docs = bool(_string_list(info_dict.get("gold_doc_ids") or info_dict.get("relevant_doc_ids")))
        if not has_answer and not has_docs:
            return 0.0

        total = 0.0
        weight = 0.0
        if has_answer and self.answer_weight > 0:
            total += self.answer_weight * score_answer(completion, answer, info_dict)
            weight += self.answer_weight
        if has_docs and self.retrieval_weight > 0:
            total += self.retrieval_weight * score_retrieval(
                completion, info_dict, top_k=self.reward_top_k
            )
            weight += self.retrieval_weight
        return total / weight if weight > 0 else 0.0


def _coerce_documents(raw: Any, *, max_documents: int, max_document_chars: int) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    values = _as_list(raw)
    for index, item in enumerate(values):
        if len(docs) >= max_documents:
            break
        if isinstance(item, str):
            doc_id = f"d{index}"
            title = ""
            text = item
            relevant = False
        elif isinstance(item, Mapping):
            doc_id = str(
                item.get("id")
                or item.get("doc_id")
                or item.get("document_id")
                or item.get("pid")
                or item.get("corpus_id")
                or f"d{index}"
            )
            title = str(item.get("title") or item.get("name") or "")
            text = str(
                item.get("text")
                or item.get("content")
                or item.get("passage")
                or item.get("document")
                or item.get("body")
                or ""
            )
            relevant = bool(item.get("relevant") or item.get("is_relevant") or item.get("gold"))
        else:
            doc_id = f"d{index}"
            title = ""
            text = str(item)
            relevant = False
        text = text[:max_document_chars] if max_document_chars > 0 else text
        if text.strip():
            docs.append(
                {
                    "id": doc_id,
                    "title": title,
                    "text": text,
                    "relevant": "true" if relevant else "false",
                }
            )
    return docs


def _coerce_gold_doc_ids(row: Mapping[str, Any], docs: list[dict[str, str]]) -> list[str]:
    for key in ("gold_doc_ids", "relevant_doc_ids", "positive_doc_ids", "evidence_doc_ids"):
        ids = _string_list(row.get(key))
        if ids:
            return ids
    qrels = row.get("qrels") or row.get("relevance") or row.get("labels")
    if isinstance(qrels, Mapping):
        ids = [str(doc_id) for doc_id, score in qrels.items() if _positive_relevance(score)]
        if ids:
            return ids
    return [doc["id"] for doc in docs if doc.get("relevant") == "true"]


def _positive_relevance(score: Any) -> bool:
    if isinstance(score, bool):
        return score
    try:
        return float(score) > 0.0
    except (TypeError, ValueError):
        return bool(score)


def _coerce_answers(row: Mapping[str, Any]) -> list[str]:
    for key in ("answers", "reference_answers", "answer", "target", "gold_answer"):
        answers = _string_list(row.get(key))
        if answers:
            return answers
    return []


def _coerce_choices(row: Mapping[str, Any]) -> Any:
    return row.get("answer_choices") or row.get("choices") or row.get("options") or {}


def canonicalize_row(
    raw_row: Mapping[str, Any],
    *,
    max_documents: int,
    max_document_chars: int,
    require_gold: bool,
    default_benchmark: str,
) -> dict[str, Any]:
    query = str(raw_row.get("query") or raw_row.get("question") or raw_row.get("prompt") or "").strip()
    if not query:
        raise ValueError(f"Reasoning-RAG row is missing query/question/prompt: keys={sorted(raw_row.keys())}")

    raw_docs = (
        raw_row.get("documents")
        or raw_row.get("docs")
        or raw_row.get("candidates")
        or raw_row.get("contexts")
        or raw_row.get("passages")
        or raw_row.get("context")
        or []
    )
    docs = _coerce_documents(raw_docs, max_documents=max_documents, max_document_chars=max_document_chars)
    gold_doc_ids = _coerce_gold_doc_ids(raw_row, docs)
    answers = _coerce_answers(raw_row)
    choices = _coerce_choices(raw_row)
    task = str(raw_row.get("task") or raw_row.get("task_type") or "rag")
    benchmark = str(raw_row.get("benchmark") or default_benchmark)
    if require_gold and not gold_doc_ids and not answers:
        raise ValueError(
            f"Reasoning-RAG row {raw_row.get('id') or raw_row.get('example_id') or '<unknown>'} "
            "has neither gold_doc_ids nor answers; refusing to create a zero-signal RL sample."
        )

    row_id = str(raw_row.get("id") or raw_row.get("example_id") or raw_row.get("qid") or "")
    if not row_id:
        row_id = f"{benchmark}-{abs(hash(query))}"

    return {
        "id": row_id,
        "benchmark": benchmark,
        "task": task,
        "query": query,
        "answers": answers,
        "answer": answers[0] if answers else "",
        "answer_choices": choices,
        "gold_doc_ids": gold_doc_ids,
        "documents": docs,
    }


def build_prompt_messages(row: Mapping[str, Any], *, system_prompt: str | None = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    docs = row.get("documents") or []
    doc_blocks: list[str] = []
    for doc in docs:
        if not isinstance(doc, Mapping):
            continue
        title = str(doc.get("title") or "").strip()
        title_line = f"\nTitle: {title}" if title else ""
        doc_blocks.append(f"[DOC {doc['id']}]{title_line}\n{doc.get('text', '')}")

    choices = row.get("answer_choices") or {}
    if isinstance(choices, Mapping) and choices:
        choices_text = "\n".join(f"{key}. {value}" for key, value in choices.items())
    elif isinstance(choices, list) and choices:
        choices_text = "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(choices))
    else:
        choices_text = ""

    user_parts = [
        f"Benchmark: {row.get('benchmark', 'reasoning-rag')}",
        f"Task: {row.get('task', 'rag')}",
        "",
        "Query:",
        str(row["query"]),
    ]
    if choices_text:
        user_parts.extend(["", "Answer choices:", choices_text])
    user_parts.extend(
        [
            "",
            "Candidate documents:",
            "\n\n".join(doc_blocks) if doc_blocks else "(no candidate documents provided)",
            "",
            "Return the final answer and supporting document IDs in the required XML tags.",
        ]
    )
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "\n".join(user_parts)})
    return messages


def _rows_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path).expanduser()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Reasoning-RAG JSONL not found: {jsonl_path}")
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{jsonl_path}:{line_no} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"Reasoning-RAG JSONL is empty: {jsonl_path}")
    return rows


def _rows_from_hf(dataset_name: str, split: str, config_name: str | None = None) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": split}
    if config_name:
        dataset = load_dataset(dataset_name, config_name, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)
    return [dict(row) for row in dataset]


def load_canonical_rows(
    *,
    source: str = "jsonl",
    local_jsonl_path: str | None = None,
    hf_dataset_name: str | None = None,
    hf_dataset_config: str | None = None,
    hf_split: str = "train",
    max_documents: int = 32,
    max_document_chars: int = 4000,
    require_gold: bool = True,
    default_benchmark: str = "reasoning-rag",
    max_examples: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[dict[str, Any]]:
    if source == "jsonl":
        if not local_jsonl_path:
            raise ValueError("source='jsonl' requires local_jsonl_path")
        raw_rows = _rows_from_jsonl(local_jsonl_path)
    elif source == "hf":
        if not hf_dataset_name:
            raise ValueError("source='hf' requires hf_dataset_name")
        raw_rows = _rows_from_hf(hf_dataset_name, hf_split, hf_dataset_config)
    else:
        raise ValueError(f"Unknown reasoning-RAG source={source!r}; expected 'jsonl' or 'hf'")

    rows = [
        canonicalize_row(
            raw_row,
            max_documents=max_documents,
            max_document_chars=max_document_chars,
            require_gold=require_gold,
            default_benchmark=default_benchmark,
        )
        for raw_row in raw_rows
    ]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if max_examples is not None and max_examples >= 0:
        rows = rows[:max_examples]
    return rows


def build_dataset_from_rows(
    rows: list[dict[str, Any]],
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> Dataset:
    dataset_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        info = {
            "id": row["id"],
            "benchmark": row["benchmark"],
            "task": row["task"],
            "answers": row["answers"],
            "answer_choices": row.get("answer_choices") or {},
            "gold_doc_ids": row["gold_doc_ids"],
            "document_count": len(row["documents"]),
            "prompt_char_count": sum(len(message["content"]) for message in build_prompt_messages(row, system_prompt=system_prompt)),
        }
        dataset_rows.append(
            {
                "example_id": index,
                "src_id": row["id"],
                "task": row["task"],
                "query": row["query"],
                "answer": row["answer"],
                "prompt": build_prompt_messages(row, system_prompt=system_prompt),
                "info": json.dumps(info, sort_keys=True),
            }
        )
    return Dataset.from_list(dataset_rows)


def _positive_int_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value >= 0 else None


def load_environment(
    *,
    source: str = "jsonl",
    local_jsonl_path: str | None = None,
    eval_jsonl_path: str | None = None,
    hf_dataset_name: str | None = None,
    hf_dataset_config: str | None = None,
    hf_split: str = "train",
    hf_eval_split: str | None = None,
    default_benchmark: str = "reasoning-rag",
    num_train_examples: int | None = None,
    num_eval_examples: int | None = 0,
    max_documents: int = 32,
    max_document_chars: int = 4000,
    require_gold: bool = True,
    reward_top_k: int = 5,
    answer_weight: float = 0.7,
    retrieval_weight: float = 0.3,
    shuffle: bool = False,
    seed: int = 0,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    **kwargs: Any,
) -> vf.Environment:
    if not math.isfinite(answer_weight) or answer_weight < 0:
        raise ValueError(f"answer_weight must be finite and >= 0, got {answer_weight}")
    if not math.isfinite(retrieval_weight) or retrieval_weight < 0:
        raise ValueError(f"retrieval_weight must be finite and >= 0, got {retrieval_weight}")

    def build_train_dataset() -> Dataset:
        rows = load_canonical_rows(
            source=source,
            local_jsonl_path=local_jsonl_path,
            hf_dataset_name=hf_dataset_name,
            hf_dataset_config=hf_dataset_config,
            hf_split=hf_split,
            max_documents=max_documents,
            max_document_chars=max_document_chars,
            require_gold=require_gold,
            default_benchmark=default_benchmark,
            max_examples=_positive_int_or_none(num_train_examples),
            seed=seed,
            shuffle=shuffle,
        )
        return build_dataset_from_rows(rows, system_prompt=system_prompt)

    def build_eval_dataset() -> Dataset:
        eval_source = source
        eval_path = eval_jsonl_path or local_jsonl_path
        eval_split = hf_eval_split or hf_split
        rows = load_canonical_rows(
            source=eval_source,
            local_jsonl_path=eval_path,
            hf_dataset_name=hf_dataset_name,
            hf_dataset_config=hf_dataset_config,
            hf_split=eval_split,
            max_documents=max_documents,
            max_document_chars=max_document_chars,
            require_gold=require_gold,
            default_benchmark=default_benchmark,
            max_examples=_positive_int_or_none(num_eval_examples),
            seed=seed + 1,
            shuffle=shuffle,
        )
        return build_dataset_from_rows(rows, system_prompt=system_prompt)

    parser = ReasoningRagParser()
    rubric = ReasoningRagRubric(
        parser=parser,
        reward_top_k=reward_top_k,
        answer_weight=answer_weight,
        retrieval_weight=retrieval_weight,
    )
    eval_dataset = build_eval_dataset if num_eval_examples is None or num_eval_examples != 0 else None
    return vf.SingleTurnEnv(
        dataset=build_train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=None,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
