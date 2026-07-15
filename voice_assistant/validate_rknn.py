#!/usr/bin/env python3
"""Load an RKNN artifact once before it replaces the active vision model."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    model = Path(sys.argv[1])
    # Downloads are staged with a .part suffix; the signed queue type establishes RKNN intent.
    if not model.is_file() or model.stat().st_size < 1024:
        raise RuntimeError("invalid RKNN artifact")
    from rknnlite.api import RKNNLite

    runtime = RKNNLite()
    try:
        if runtime.load_rknn(str(model)) != 0:
            raise RuntimeError("RKNN load failed")
        if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError("RKNN NPU initialization failed")
    finally:
        runtime.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
