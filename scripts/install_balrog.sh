#!/usr/bin/env bash
# Install BALROG dependencies for the BabyAI environment.
#
# The `balrog` package on PyPI is Paylogic's ACL library (wrong package).
# The correct BALROG (Benchmarking Agentic LLM and VLM Reasoning On Games)
# must be installed from GitHub with --no-deps to avoid pulling google-genai
# which conflicts with verifiers -> prime-sandboxes.
set -euo pipefail

echo "==> Installing balrog dependency group (textworld, omegaconf, gym)..."
uv sync --group balrog

echo "==> Installing Pillow (MiniGrid rendering requires PIL)..."
uv pip install Pillow

echo "==> Installing balrog from GitHub (--no-deps)..."
uv pip install "git+https://github.com/DavidePaglieri/BALROG.git" --no-deps

echo "==> Verifying imports..."
uv run python -c "
from balrog.environments import make_env
from balrog.prompt_builder import HistoryPromptBuilder
import balrog_bench
print('All BALROG imports OK')
"

echo "==> Done."
