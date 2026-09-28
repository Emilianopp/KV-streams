# Reproducing the 62k TextWorld dataset

Run `bash setup.sh` first, then generate the canonical dataset on Linux. It
contains 61,744 training games and a fixed 256-game evaluation split across
`tw-cooking`, `tw-simple`, `tw-coin_collector`, and `tw-treasure_hunter`.

```bash
source .venv/bin/activate
data_root="${DATA_ROOT:-$PWD/data}"
base="$data_root/textworld_all4_uniform_total12000_train11744_eval256_seed20260719"
extra="$data_root/textworld_all4_uniform_add50000_train50000_eval0_seed20270719"
final="$data_root/textworld_all4_uniform_total62000_train61744_eval256_seed20260720"

python experiments/textworld_env/generate_dataset_all_families_parallel.py \
  --output "$base" --total-train 11744 --total-eval 256 \
  --seed 20260719 --workers 40 --retries 10 --task-timeout 120

python experiments/textworld_env/generate_dataset_all_families_parallel.py \
  --output "$extra" --total-train 50000 --total-eval 0 \
  --seed 20270719 --workers 40 --retries 10 --task-timeout 120

python experiments/textworld_env/merge_textworld_datasets.py \
  --base "$base" --additional "$extra" --output "$final"
```

The merge keeps the original evaluation games unchanged. Each family has the
same split:

| Split | Easy | Medium | Hard | Per family | Total |
|---|---:|---:|---:|---:|---:|
| Train | 5,146 | 5,146 | 5,144 | 15,436 | 61,744 |
| Eval | 22 | 21 | 21 | 64 | 256 |

Keep every same-stem `.z8`, `.json`, `.result.json`, and `.ni` file together;
missing `.json` sidecars disables TextWorld score, max-score, and win reporting.
The final tree should contain 62,000 `.z8`, 62,000 `.ni`, and 124,000 `.json`
files under `games/`, plus `dataset/`, `eval_dataset/`, `metadata.json`, and
`manifest.jsonl`.

Paper runs: every config in `configs/textworld/` points at
`__TEXTWORLD_DATASET__`, which in the paper runs was
`textworld_all4_total60000_train59744_eval256_seed20260719` (59,744 train /
256 eval games across the same four families). Substitute your dataset path
before launching — see the arm table in `README.md`.
