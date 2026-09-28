from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Mapping

PROJECT_SLUG = "balrog-crafter-primerl"


def scratch_dir(
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    exists_fn: Callable[[Path], bool] | None = None,
) -> Path:
    """Resolve the scratch root used by local and Mila launcher scripts."""

    env = os.environ if env is None else env
    override = env.get("BALROG_CRAFTER_SCRATCH_DIR")
    if override:
        return Path(override).expanduser()

    user = env.get("USER") or env.get("LOGNAME")
    if user:
        network_user_root = Path("/network/scratch") / user[0] / user
        exists = Path.exists if exists_fn is None else exists_fn
        if exists(network_user_root):
            return network_user_root / PROJECT_SLUG

    return (Path.home() if home is None else home) / "scratch" / PROJECT_SLUG


def project_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    override = env.get("BALROG_CRAFTER_PROJECT_DIR")
    if override:
        return Path(override).expanduser()
    return Path.cwd()


def model_label(model_path: str) -> str:
    label = Path(model_path.rstrip("/")).name or model_path
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return label.lower() or "model"
