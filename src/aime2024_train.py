from pathlib import Path

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.utils.data_utils import extract_boxed_answer


DEFAULT_INSTRUCTION_PROMPT_PRE = (
    "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
)
DEFAULT_INSTRUCTION_PROMPT_POST = ""
DATASET_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
DEFAULT_CACHED_ARROW = Path(
    "/network/scratch/d/dane.malenfant/.cache/datasets/"
    "HuggingFaceH4___aime_2024/default/0.0.0/"
    f"{DATASET_REVISION}/aime_2024-train.arrow"
)


def _load_raw_aime(cache_arrow_path: str | None = None) -> Dataset:
    arrow_path = Path(cache_arrow_path) if cache_arrow_path else DEFAULT_CACHED_ARROW
    if arrow_path.exists():
        return Dataset.from_file(str(arrow_path))
    return load_dataset(
        "HuggingFaceH4/aime_2024",
        split="train",
        revision=DATASET_REVISION,
        trust_remote_code=False,
    )


def _build_dataset(
    instruction_prompt_pre: str,
    instruction_prompt_post: str,
    cache_arrow_path: str | None,
) -> Dataset:
    raw = _load_raw_aime(cache_arrow_path)
    rows = [
        {
            "question": instruction_prompt_pre + row["problem"] + instruction_prompt_post,
            "answer": str(int(row["answer"])),
        }
        for row in raw
    ]
    return Dataset.from_list(rows)


def load_environment(
    system_prompt: str | None = None,
    instruction_prompt_pre: str = DEFAULT_INSTRUCTION_PROMPT_PRE,
    instruction_prompt_post: str = DEFAULT_INSTRUCTION_PROMPT_POST,
    cache_arrow_path: str | None = None,
    **kwargs,
) -> vf.Environment:
    def build_dataset() -> Dataset:
        return _build_dataset(instruction_prompt_pre, instruction_prompt_post, cache_arrow_path)

    parser = vf.MaybeThinkParser(extract_boxed_answer)
    rubric = vf.MathRubric(parser=parser)
    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
