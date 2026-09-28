#!/usr/bin/env python3
"""Run verifiers eval with kv_eviction hooks installed first."""

import kv_eviction  # noqa: F401
from verifiers.scripts.eval import main


if __name__ == "__main__":
    main()
